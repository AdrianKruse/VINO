// Point rendering (round, world-sized splats with depth fade) adapted from ARROW
// (https://ifradlin.github.io/arrow/, MIT).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Names are only tooltips; the cards show thumbnails. (No all-digit ids below 2^32: those would jump the order.)
const SCENES = {
    scene0024_00: 'ScanNet 1',
    scene0618_00: 'ScanNet 2',
    '5942004064': 'ScanNet++ 1',
    '09c1414f1b': 'ScanNet++ 2',
    d240136ce4_noceil: 'ScanNet++ 3',
};
// Training (backbone), ordered so that each step changes one thing
const MODELS = {
    sonata: 'Sonata (PTv3)',
    a2: 'Sonata (Volt)',
    vino: 'VINO (Volt)',
};
const MODES = { pca: 'PCA', sim: 'Similarity' };
const HEAT = [[0.12, 0.10, 0.20], [0.32, 0.07, 0.50], [0.72, 0.21, 0.47], [0.99, 0.55, 0.38], [0.99, 0.99, 0.75]];

const root = document.getElementById('viewer');
const canvasHost = root.querySelector('.viewer-canvas');
const status = root.querySelector('.viewer-status');
const BACKGROUND = new THREE.Color(getComputedStyle(canvasHost).getPropertyValue('--fade').trim());

const state = { scene: Object.keys(SCENES)[0], model: 'vino', mode: 'pca', query: null, split: 0.5 };
const cache = new Map();
let current = null;

// No MSAA: it does not smooth the round splats (their edge is a discard), only the marker
const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
renderer.setClearColor(0x000000, 0);
canvasHost.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const scene3d = new THREE.Scene();
const material = new THREE.ShaderMaterial({
    uniforms: {
        size: { value: 0.03 },
        viewportHeight: { value: 1 },
        fadeNear: { value: 1 },
        fadeFar: { value: 10 },
        splitX: { value: 0 },
        background: { value: BACKGROUND },
    },
    vertexShader: `
        uniform float size;
        uniform float viewportHeight;
        uniform float fadeNear;
        uniform float fadeFar;
        attribute vec3 rgb;
        attribute vec3 color;  // features: PCA or similarity
        attribute float relativeSize;  // uint8 / 255, exported as 100 x the point's spacing over the median
        varying vec3 vRgb;
        varying vec3 vColor;
        varying float vFade;
        void main() {
            vec4 world = modelMatrix * vec4(position, 1.0);
            if (dot(normal, cameraPosition - world.xyz) < 0.0) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }
            vec4 mv = viewMatrix * world;
            gl_Position = projectionMatrix * mv;
            float pointSize = size * relativeSize * 2.55;
            gl_PointSize = max(1.5, pointSize * projectionMatrix[1][1] * 0.5 * viewportHeight / -mv.z);
            vRgb = rgb;
            vColor = color;
            vFade = smoothstep(fadeNear, fadeFar, -mv.z) * 0.3;
        }`,
    fragmentShader: `
        uniform vec3 background;
        uniform float splitX;  // drawing-buffer pixels: RGB to the left, features to the right
        varying vec3 vRgb;
        varying vec3 vColor;
        varying float vFade;
        void main() {
            vec2 c = gl_PointCoord * 2.0 - 1.0;
            float d = dot(c, c);
            if (d > 1.0) discard;
            vec3 base = gl_FragCoord.x < splitX ? vRgb : vColor;
            vec3 shaded = base * (0.75 + 0.25 * sqrt(1.0 - d));
            gl_FragColor = vec4(mix(shaded, background, vFade), 1.0);
        }`,
});
const marker = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0x5eead4, depthTest: false }),
);
marker.renderOrder = 1;
marker.visible = false;
scene3d.add(marker);

// RGB left of the split, features right of it. The handle is a DOM overlay, so dragging it never orbits.
const split = document.createElement('div');
split.className = 'viewer-split';
split.innerHTML = '<span class="viewer-split-label">RGB</span><span class="viewer-split-knob"></span>'
    + '<span class="viewer-split-label viewer-split-mode"></span>';
canvasHost.appendChild(split);
split.addEventListener('pointerdown', (e) => {
    split.setPointerCapture(e.pointerId);
    e.preventDefault();
});
split.addEventListener('pointermove', (e) => {
    if (!split.hasPointerCapture(e.pointerId)) return;
    const rect = canvasHost.getBoundingClientRect();
    state.split = Math.min(0.97, Math.max(0.03, (e.clientX - rect.left) / rect.width));
    placeSplit();
    requestRender();
});

function placeSplit() {
    split.style.left = `${state.split * 100}%`;
    split.querySelector('.viewer-split-mode').textContent = MODES[state.mode];
}

async function fetchBin(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return response.arrayBuffer();
}

async function loadScene(id) {
    if (cache.has(id)) return cache.get(id);
    const base = `assets/viewer/${id}/`;
    const meta = await (await fetch(base + 'scene.json')).json();
    const buffer = await fetchBin(base + 'points.bin');
    const n = meta.points;
    const q = new Uint16Array(buffer, 0, n * 3);
    const lo = meta.bbox_min, hi = meta.bbox_max;
    const center = lo.map((v, i) => (v + hi[i]) / 2);
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
        const a = i % 3;
        positions[i] = lo[a] + (q[i] / 65535) * (hi[a] - lo[a]) - center[a];
    }
    const entry = {
        meta,
        base,
        positions,
        rgb: new Uint8Array(buffer, n * 6, n * 3),
        normals: new Int8Array(buffer, n * 9, n * 3),
        sizes: new Uint8Array(buffer, n * 12, n),
        radius: Math.hypot(...hi.map((v, i) => v - lo[i])) / 2,
        models: {},
    };
    cache.set(id, entry);
    return entry;
}

async function loadModel(entry, key) {
    if (entry.models[key]) return entry.models[key];
    const { tokens: t, dims: k } = entry.meta.models[key];
    const n = entry.meta.points;
    const buffer = await fetchBin(entry.base + key + '.bin');
    const scale = new Float32Array(buffer, 0, k);
    const p2t = new Uint16Array(buffer, k * 4, n);
    const pca = new Uint8Array(buffer, k * 4 + n * 2, t * 3);
    const feat = new Int8Array(buffer, k * 4 + n * 2 + t * 3, t * k);
    const weight = scale.map((s) => s * s);
    const norm = new Float32Array(t);
    for (let j = 0; j < t; j++) {
        let s = 0;
        for (let i = 0; i < k; i++) s += feat[j * k + i] * feat[j * k + i] * weight[i];
        norm[j] = Math.sqrt(s);
    }
    entry.models[key] = { t, k, p2t, pca, feat, weight, norm };
    return entry.models[key];
}

function heat(t) {
    const x = t * (HEAT.length - 1);
    const i = Math.min(Math.floor(x), HEAT.length - 2);
    const f = x - i;
    return HEAT[i].map((v, c) => v + (HEAT[i + 1][c] - v) * f);
}

function similarityColors(entry, model, query) {
    const { t, k, feat, weight, norm, p2t } = model;
    const qt = p2t[query];
    const qv = new Float32Array(k);
    for (let i = 0; i < k; i++) qv[i] = feat[qt * k + i] * weight[i];
    const sims = new Float32Array(t);
    for (let j = 0; j < t; j++) {
        let s = 0;
        for (let i = 0; i < k; i++) s += feat[j * k + i] * qv[i];
        sims[j] = s / (norm[j] * norm[qt]);
    }
    const median = Float32Array.from(sims).sort()[t >> 1];
    const tokenColor = new Uint8Array(t * 3);
    for (let j = 0; j < t; j++) {
        const v = Math.min(1, Math.max(0, (sims[j] - median) / (1 - median)));
        const c = heat(v);
        for (let a = 0; a < 3; a++) tokenColor[j * 3 + a] = Math.round(c[a] * 255);
    }
    return perPoint(tokenColor, p2t);
}

function perPoint(tokenColor, p2t) {
    const out = new Uint8Array(p2t.length * 3);
    for (let i = 0; i < p2t.length; i++) {
        const j = p2t[i] * 3;
        out[i * 3] = tokenColor[j];
        out[i * 3 + 1] = tokenColor[j + 1];
        out[i * 3 + 2] = tokenColor[j + 2];
    }
    return out;
}

function setColors(colors) {
    const attribute = current.geometry.getAttribute('color');  // reuse the GPU buffer
    attribute.array.set(colors);
    attribute.needsUpdate = true;
    requestRender();
}

async function update() {
    const entry = await loadScene(state.scene);
    if (!current || current.entry !== entry) {
        if (current) {
            scene3d.remove(current.points);
            current.geometry.dispose();
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(entry.positions, 3));
        geometry.setAttribute('normal', new THREE.BufferAttribute(entry.normals, 3, true));
        geometry.setAttribute('relativeSize', new THREE.BufferAttribute(entry.sizes, 1, true));
        geometry.setAttribute('rgb', new THREE.BufferAttribute(entry.rgb, 3, true));
        geometry.setAttribute('color', new THREE.BufferAttribute(entry.rgb.slice(), 3, true));  // RGB until features load
        const points = new THREE.Points(geometry, material);
        scene3d.add(points);
        current = { entry, geometry, points };
        resetView();
    }
    status.textContent = 'Loading features…';
    const model = await loadModel(entry, state.model);
    if (state.mode === 'pca') setColors(perPoint(model.pca, model.p2t));
    else setColors(similarityColors(entry, model, state.query));
    showStatus();
}

function resetView() {
    const r = current.entry.radius;
    const d = r * Math.max(1, 1.4 / camera.aspect);
    camera.position.set(-0.9 * d, -1.3 * d, 1.2 * d);
    controls.target.set(0, 0, 0);
    controls.update();
    material.uniforms.size.value = 1.5 * current.entry.meta.spacing;
}

function showStatus() {
    status.textContent = state.mode === 'sim'
        ? `Cosine similarity of ${MODELS[state.model]} features to the selected point.`
        : 'Drag the handle to compare with RGB. Click any point to see which parts of the scene have similar features.';
}

function buttons(group, options, key) {
    const host = root.querySelector(`[data-group="${group}"]`);
    for (const [value, label] of Object.entries(options)) {
        const button = document.createElement('button');
        if (group === 'scene') {
            const thumb = document.createElement('img');
            thumb.src = `assets/viewer/${value}/thumb.webp`;
            thumb.alt = label;
            button.title = label;
            button.append(thumb);
        } else {
            button.append(label);
        }
        button.dataset.value = value;
        button.addEventListener('click', () => {
            if (value === 'sim' && state.query === null) return;
            if (key === 'scene') {
                state.query = null;
                marker.visible = false;
                if (state.mode === 'sim') state.mode = 'pca';
            }
            state[key] = value;
            refreshButtons();
            update().catch(fail);
        });
        host.appendChild(button);
    }
}

function refreshButtons() {
    for (const button of root.querySelectorAll('[data-group] button')) {
        const key = button.parentElement.dataset.group;
        button.classList.toggle('active', state[key] === button.dataset.value);
        button.disabled = button.dataset.value === 'sim' && state.query === null;
    }
    placeSplit();
}

function fail(error) {
    status.textContent = `Could not load the scene (${error.message}).`;
}

const raycaster = new THREE.Raycaster();
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || !current || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(ndc, camera);
    raycaster.params.Points.threshold = material.uniforms.size.value;
    const normals = current.entry.normals;
    const toCamera = new THREE.Vector3();
    const hit = raycaster.intersectObject(current.points).find(({ index, point }) => {
        toCamera.subVectors(camera.position, point);
        return normals[index * 3] * toCamera.x + normals[index * 3 + 1] * toCamera.y + normals[index * 3 + 2] * toCamera.z >= 0;
    });
    if (!hit) return;
    state.query = hit.index;
    state.mode = 'sim';
    marker.position.copy(hit.point);
    marker.scale.setScalar(current.entry.radius * 0.012);
    marker.visible = true;
    refreshButtons();
    update().catch(fail);
});

new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = canvasHost;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));  // 3x phone screens: 2.25x the pixels for little gain
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    material.uniforms.viewportHeight.value = h * renderer.getPixelRatio();
    requestRender();
}).observe(canvasHost);

// Draw only when something changed. While the camera moves (also while damping settles) controls.update()
// fires 'change', which asks for the next frame.
let frame = null;
function requestRender() {
    if (frame === null) frame = requestAnimationFrame(render);
}
function render() {
    frame = null;
    controls.update();
    if (current) {  // fade only the far part of the scene, however far the camera is zoomed out
        const d = camera.position.distanceTo(controls.target);
        material.uniforms.fadeNear.value = d - 0.5 * current.entry.radius;
        material.uniforms.fadeFar.value = d + current.entry.radius;
    }
    material.uniforms.splitX.value = state.split * renderer.domElement.width;
    renderer.render(scene3d, camera);
}
controls.addEventListener('change', requestRender);

buttons('scene', SCENES, 'scene');
buttons('model', MODELS, 'model');
buttons('mode', MODES, 'mode');
refreshButtons();
showStatus();

new IntersectionObserver((entries, observer) => {
    if (entries[0].isIntersecting) {
        observer.disconnect();
        update().catch(fail);
    }
}).observe(root);

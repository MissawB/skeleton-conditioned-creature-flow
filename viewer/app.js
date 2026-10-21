// viewer/app.js
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

let scene, camera, renderer, controls;
let pointCloud, skeletonGroup, floorGroup;
let creaturesData = [];
let currentCreature = null;
let currentIdx = -1;
let isPlaying = false;
let flowT = 0.0;

const NOISE_COLOR = new THREE.Color(0x8da197);
const BONE_COLOR = new THREE.Color(0xe9e3d2);
const OCHRE = 0xd6a23e;
const BACKGROUND = 0x141c1a;
const PLAY_SPEED = 0.25; // flow time per second
const DEFAULT_CAM_OFFSET = new THREE.Vector3(1.9, 0.9, 2.3);

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);

function init() {
  const container = $("canvas-container");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(BACKGROUND);
  scene.fog = new THREE.Fog(BACKGROUND, 3.0, 7.5);

  camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.05, 100);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 0.8;
  controls.maxDistance = 8;

  floorGroup = new THREE.Group();
  scene.add(floorGroup);
  buildFloor();

  skeletonGroup = new THREE.Group();
  skeletonGroup.renderOrder = 2;
  scene.add(skeletonGroup);

  setupUI();
  loadData();
  animate();

  window.addEventListener("resize", onWindowResize);
}

// Round, soft-edged sprite so points read as particles instead of squares
function makeDotTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.55, "rgba(255,255,255,0.9)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

function buildFloor() {
  // Faint concentric rings, like the base of a specimen jar
  const ringMat = new THREE.LineBasicMaterial({ color: 0x34423f, transparent: true, opacity: 0.55 });
  for (let r = 0.35; r <= 1.05; r += 0.35) {
    const pts = [];
    for (let a = 0; a <= 96; a++) {
      const th = (a / 96) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(th) * r, 0, Math.sin(th) * r));
    }
    floorGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
  }

  // Soft contact shadow under the specimen
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(0,0,0,0.35)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 1.0),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.001;
  floorGroup.add(shadow);
}

function setupUI() {
  const slider = $("time-slider");
  const playBtn = $("play-btn");
  const skelBtn = $("toggle-skeleton-btn");
  const resetBtn = $("reset-cam-btn");

  slider.addEventListener("input", (e) => {
    setPlaying(false);
    setFlowTime(parseFloat(e.target.value));
  });

  playBtn.addEventListener("click", () => {
    if (!isPlaying && flowT >= 1.0) setFlowTime(0.0);
    setPlaying(!isPlaying);
  });

  skelBtn.addEventListener("click", () => {
    skeletonGroup.visible = !skeletonGroup.visible;
    skelBtn.setAttribute("aria-pressed", String(skeletonGroup.visible));
  });

  resetBtn.addEventListener("click", resetCamera);

  window.addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("button, input")) {
      if (e.key === " " || e.key.startsWith("Arrow")) return;
    }
    if (e.key === " ") {
      e.preventDefault();
      playBtn.click();
    } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      setPlaying(false);
      setFlowTime(flowT + (e.key === "ArrowRight" ? 0.05 : -0.05));
    } else if (e.key === "s") {
      skelBtn.click();
    } else if (/^[1-9]$/.test(e.key)) {
      loadCreature(parseInt(e.key, 10) - 1);
    }
  });
}

function setPlaying(on) {
  isPlaying = on;
  const btn = $("play-btn");
  btn.classList.toggle("is-playing", on);
  btn.setAttribute("aria-label", on ? "Pause flow" : "Play flow");
}

function setFlowTime(t) {
  flowT = Math.min(1, Math.max(0, t));
  const slider = $("time-slider");
  slider.value = flowT;
  slider.style.setProperty("--fill", `${flowT * 100}%`);
  $("time-val").textContent = flowT.toFixed(2);

  if (currentCreature) {
    const steps = currentCreature.trajectory.length - 1;
    $("step-val").textContent = `step ${Math.round(flowT * steps)} of ${steps}`;
    document.querySelectorAll("#ticks span").forEach((tick, i) => {
      tick.classList.toggle("passed", i / steps <= flowT + 1e-6);
    });
  }
  updatePointCloud(flowT);
}

function setStatus(html) {
  $("status").innerHTML = html;
}

// "Lion (Felidae)" -> ["Lion", "Felidae"]
function splitName(name) {
  const m = name.match(/^(.*?)\s*\((.*)\)\s*$/);
  return m ? [m[1], m[2]] : [name, ""];
}

function loadData() {
  fetch("data/creatures.json")
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((data) => {
      creaturesData = data.creatures;
      buildSpecimenList();
      loadCreature(0);
      if (!reducedMotion) {
        setFlowTime(0.0);
        setPlaying(true);
      }
    })
    .catch((err) => {
      console.warn("Could not load creatures.json:", err);
      $("specimen-name").textContent = "No specimens";
      setStatus(
        "No trajectories found in <code>viewer/data/creatures.json</code>. " +
        "Run <code>python scripts/sample_and_export.py</code>, then reload this page."
      );
    });
}

function buildSpecimenList() {
  const list = $("specimen-list");
  list.innerHTML = "";
  creaturesData.forEach((c, i) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    const [common] = splitName(c.name);
    btn.innerHTML = `<span class="label"></span><span class="key">${i + 1}</span>`;
    btn.querySelector(".label").textContent = common;
    btn.title = c.name;
    btn.addEventListener("click", () => loadCreature(i));
    li.appendChild(btn);
    list.appendChild(li);
  });
}

function buildTicks(steps) {
  const ticks = $("ticks");
  ticks.innerHTML = "";
  for (let i = 0; i <= steps; i++) {
    const tick = document.createElement("span");
    tick.style.left = `${(i / steps) * 100}%`;
    if (i % 5 === 0) tick.classList.add("major");
    ticks.appendChild(tick);
  }
}

function loadCreature(idx) {
  if (!creaturesData[idx]) return;
  currentIdx = idx;
  currentCreature = creaturesData[idx];

  const [common, family] = splitName(currentCreature.name);
  $("specimen-name").textContent = common;
  $("specimen-family").textContent =
    currentCreature.type === "hybrid" ? `Hybrid: ${family}` : family;
  document.querySelectorAll("#specimen-list button").forEach((b, i) => {
    b.setAttribute("aria-current", String(i === idx));
  });

  const traj = currentCreature.trajectory;
  buildTicks(traj.length - 1);
  $("point-count").textContent = traj[0].length;
  $("joint-count").textContent = currentCreature.joints.length;

  // Rest the floor under the final shape and aim the camera at its centre
  const final = traj[traj.length - 1];
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  final.forEach((p) => {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k];
      if (p[k] > max[k]) max[k] = p[k];
    }
  });
  floorGroup.position.set((min[0] + max[0]) / 2, min[1] - 0.01, (min[2] + max[2]) / 2);
  currentCreature._center = new THREE.Vector3(
    (min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2
  );
  if (!camera.userData.placed) {
    resetCamera();
    camera.userData.placed = true;
  }

  buildSkeleton();
  buildPointCloud(final.length);
  setFlowTime(flowT);
}

function resetCamera() {
  const c = currentCreature && currentCreature._center ? currentCreature._center : new THREE.Vector3();
  // Back off on portrait screens so the whole body fits the narrow frustum
  const fit = Math.max(1, 0.8 / camera.aspect);
  controls.target.copy(c);
  camera.position.copy(c).addScaledVector(DEFAULT_CAM_OFFSET, fit);
  const dist = DEFAULT_CAM_OFFSET.length() * fit;
  scene.fog.near = dist;
  scene.fog.far = dist * 2.4;
  controls.update();
}

function buildSkeleton() {
  while (skeletonGroup.children.length > 0) {
    const child = skeletonGroup.children[0];
    child.geometry.dispose();
    skeletonGroup.remove(child);
  }

  const joints = currentCreature.joints;
  const parents = currentCreature.joint_parents;

  // Drawn through the cloud so the conditioning stays visible at t = 1
  const jointMat = new THREE.MeshBasicMaterial({ color: OCHRE, depthTest: false, transparent: true });
  const boneMat = new THREE.LineBasicMaterial({ color: OCHRE, depthTest: false, transparent: true, opacity: 0.75 });

  joints.forEach((j) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 10), jointMat);
    m.position.set(j[0], j[1], j[2]);
    m.renderOrder = 2;
    skeletonGroup.add(m);
  });

  const segs = [];
  for (let c = 1; c < joints.length; c++) {
    const p = parents[c];
    if (p >= 0) segs.push(...joints[p], ...joints[c]);
  }
  const boneGeo = new THREE.BufferGeometry();
  boneGeo.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
  const bones = new THREE.LineSegments(boneGeo, boneMat);
  bones.renderOrder = 2;
  skeletonGroup.add(bones);
}

function buildPointCloud(numPoints) {
  if (pointCloud && pointCloud.geometry.attributes.position.count === numPoints) return;
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(numPoints * 3), 3));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(numPoints * 3), 3));
  const mat = new THREE.PointsMaterial({
    size: 0.03,
    map: makeDotTexture(),
    vertexColors: true,
    transparent: true,
    alphaTest: 0.2,
    depthWrite: true,
  });
  pointCloud = new THREE.Points(geo, mat);
  pointCloud.frustumCulled = false;
  scene.add(pointCloud);
}

function updatePointCloud(tVal) {
  if (!currentCreature || !pointCloud) return;
  const traj = currentCreature.trajectory;
  const numSteps = traj.length - 1;

  const stepFloat = tVal * numSteps;
  const stepIdx = Math.min(numSteps - 1, Math.floor(stepFloat));
  const alpha = stepFloat - stepIdx;
  const ptsA = traj[stepIdx];
  const ptsB = traj[stepIdx + 1];

  const pos = pointCloud.geometry.attributes.position;
  const col = pointCloud.geometry.attributes.color;
  const floorY = floorGroup.position.y;
  const tmp = new THREE.Color();

  for (let i = 0; i < ptsA.length; i++) {
    const x = (1 - alpha) * ptsA[i][0] + alpha * ptsB[i][0];
    const y = (1 - alpha) * ptsA[i][1] + alpha * ptsB[i][1];
    const z = (1 - alpha) * ptsA[i][2] + alpha * ptsB[i][2];
    pos.array[i * 3] = x;
    pos.array[i * 3 + 1] = y;
    pos.array[i * 3 + 2] = z;

    // Colour tracks flow time (sage noise -> bone), with light falling from above
    const shade = 0.72 + 0.28 * Math.min(1, Math.max(0, (y - floorY) / 1.1));
    tmp.copy(NOISE_COLOR).lerp(BONE_COLOR, tVal).multiplyScalar(shade);
    col.array[i * 3] = tmp.r;
    col.array[i * 3 + 1] = tmp.g;
    col.array[i * 3 + 2] = tmp.b;
  }
  pos.needsUpdate = true;
  col.needsUpdate = true;
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

let lastFrame = performance.now();
function animate(now = performance.now()) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;

  if (isPlaying) {
    const next = flowT + dt * PLAY_SPEED;
    setFlowTime(next);
    if (next >= 1.0) setPlaying(false);
  }

  controls.update();
  renderer.render(scene, camera);
}

// Modules run after the document is parsed, so the DOM is ready here
init();

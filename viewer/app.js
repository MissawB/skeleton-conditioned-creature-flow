// viewer/app.js
let scene, camera, renderer, controls;
let pointCloud, skeletonGroup;
let creaturesData = [];
let currentCreature = null;
let isPlaying = false;
let playProgress = 0.0;

function init() {
  const container = document.getElementById("canvas-container");
  
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0f1d);
  scene.fog = new THREE.FogExp2(0x0a0f1d, 0.25);

  // Camera
  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(2.0, 1.5, 2.5);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);

  // Lights & Grid
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
  dirLight.position.set(5, 10, 5);
  scene.add(dirLight);

  const grid = new THREE.GridHelper(4, 20, 0x1e293b, 0x0f172a);
  grid.position.y = -0.37;
  scene.add(grid);

  skeletonGroup = new THREE.Group();
  scene.add(skeletonGroup);

  setupUI();
  loadData();
  animate();

  window.addEventListener("resize", onWindowResize);
}

function setupUI() {
  const slider = document.getElementById("time-slider");
  const timeVal = document.getElementById("time-val");
  const playBtn = document.getElementById("play-btn");
  const select = document.getElementById("creature-select");
  const skelBtn = document.getElementById("toggle-skeleton-btn");
  const resetBtn = document.getElementById("reset-cam-btn");

  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    timeVal.textContent = val.toFixed(2);
    updatePointCloud(val);
  });

  playBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? "⏸ Pause Flow" : "▶ Play Flow";
  });

  select.addEventListener("change", (e) => {
    const idx = parseInt(e.target.value);
    loadCreature(idx);
  });

  skelBtn.addEventListener("click", () => {
    skeletonGroup.visible = !skeletonGroup.visible;
  });

  resetBtn.addEventListener("click", () => {
    camera.position.set(2.0, 1.5, 2.5);
    controls.target.set(0, 0, 0);
  });
}

function loadData() {
  const select = document.getElementById("creature-select");
  const statusLabel = document.getElementById("status-label");

  fetch("data/creatures.json")
    .then((res) => {
      if (!res.ok) throw new Error("creatures.json not found");
      return res.json();
    })
    .then((data) => {
      creaturesData = data.creatures;
      select.innerHTML = "";
      creaturesData.forEach((c, i) => {
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = c.name;
        select.appendChild(opt);
      });
      loadCreature(0);
      statusLabel.textContent = "Data Loaded";
    })
    .catch((err) => {
      console.warn("Using procedural fallback:", err);
      statusLabel.textContent = "Offline Mode (Run sample_and_export.py)";
    });
}

function loadCreature(idx) {
  if (!creaturesData[idx]) return;
  currentCreature = creaturesData[idx];
  
  // Rebuild Skeleton
  while (skeletonGroup.children.length > 0) {
    skeletonGroup.remove(skeletonGroup.children[0]);
  }

  const joints = currentCreature.joints;
  const parents = currentCreature.joint_parents;
  const jointGeo = new THREE.SphereGeometry(0.015, 12, 12);
  const jointMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
  const boneMat = new THREE.LineBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.6 });

  joints.forEach((j) => {
    const m = new THREE.Mesh(jointGeo, jointMat);
    m.position.set(j[0], j[1], j[2]);
    skeletonGroup.add(m);
  });

  for (let c = 1; c < joints.length; c++) {
    const p = parents[c];
    if (p >= 0) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(joints[p][0], joints[p][1], joints[p][2]),
        new THREE.Vector3(joints[c][0], joints[c][1], joints[c][2])
      ]);
      const line = new THREE.Line(geo, boneMat);
      skeletonGroup.add(line);
    }
  }

  updatePointCloud(parseFloat(document.getElementById("time-slider").value));
}

function updatePointCloud(tVal) {
  if (!currentCreature || !currentCreature.trajectory) return;
  const traj = currentCreature.trajectory;
  const numSteps = traj.length - 1;
  
  const stepFloat = tVal * numSteps;
  const stepIdx = Math.min(numSteps - 1, Math.floor(stepFloat));
  const alpha = stepFloat - stepIdx;
  
  const ptsA = traj[stepIdx];
  const ptsB = traj[Math.min(numSteps, stepIdx + 1)];
  const numPoints = ptsA.length;
  
  document.getElementById("point-count").textContent = numPoints;

  const positions = new Float32Array(numPoints * 3);
  const colors = new Float32Array(numPoints * 3);

  for (let i = 0; i < numPoints; i++) {
    const x = (1 - alpha) * ptsA[i][0] + alpha * ptsB[i][0];
    const y = (1 - alpha) * ptsA[i][1] + alpha * ptsB[i][1];
    const z = (1 - alpha) * ptsA[i][2] + alpha * ptsB[i][2];

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    // Gradient based on elevation and flow progression
    colors[i * 3] = 0.2 + 0.6 * (y + 0.5);
    colors[i * 3 + 1] = 0.5 + 0.4 * tVal;
    colors[i * 3 + 2] = 0.9 - 0.3 * (y + 0.5);
  }

  if (pointCloud) scene.remove(pointCloud);

  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  pGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const pMat = new THREE.PointsMaterial({
    size: 0.016,
    vertexColors: true,
    transparent: true,
    opacity: 0.88
  });

  pointCloud = new THREE.Points(pGeo, pMat);
  scene.add(pointCloud);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  if (isPlaying) {
    playProgress += 0.005;
    if (playProgress > 1.0) playProgress = 0.0;
    document.getElementById("time-slider").value = playProgress;
    document.getElementById("time-val").textContent = playProgress.toFixed(2);
    updatePointCloud(playProgress);
  }

  controls.update();
  renderer.render(scene, camera);
}

window.onload = init;

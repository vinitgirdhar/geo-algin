import * as THREE from 'three';

const DEPTH = 720;        // how deep the starfield extends behind the camera
const SPREAD_X = 340;     // half-width of the field
const SPREAD_Y = 220;     // half-height of the field

export function createScene({ canvas, config, isMobile, reduced }) {
  const colorA = new THREE.Color(config.nebula.colorA); // magenta
  const colorB = new THREE.Color(config.nebula.colorB); // cyan

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(0x050507, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile ? 1.5 : 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    60, window.innerWidth / window.innerHeight, 0.1, 1500
  );
  camera.position.set(0, 0, 0);

  /* Targets set from App.jsx; render() eases toward them every
     frame so everything inherits the same weighted, drifting feel. */
  const target = { scroll: 0, focal: 0, fov: 60, px: 0, py: 0 };
  const current = { scroll: 0, focal: 0, fov: 60, px: 0, py: 0 };

  /* ---------------- 1. Starfield ---------------- */

  const starCount = isMobile ? config.stars.mobile : config.stars.desktop;
  const positions = new Float32Array(starCount * 3);
  const sizes = new Float32Array(starCount);
  const shades = new Float32Array(starCount);  // 0..1 → magenta-ish .. cyan-ish tint
  const phases = new Float32Array(starCount);  // twinkle offset

  for (let i = 0; i < starCount; i++) {
    positions[i * 3 + 0] = (Math.random() * 2 - 1) * SPREAD_X;
    positions[i * 3 + 1] = (Math.random() * 2 - 1) * SPREAD_Y;
    positions[i * 3 + 2] = Math.random() * DEPTH; // stored 0..DEPTH, shader maps to -DEPTH..0
    // mostly faint stars, a few bright ones
    sizes[i] = Math.random() < 0.06
      ? 1.8 + Math.random() * 1.6
      : 0.5 + Math.random() * 1.2;
    shades[i] = Math.random();
    phases[i] = Math.random();
  }

  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  starGeo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
  starGeo.setAttribute('aShade', new THREE.BufferAttribute(shades, 1));
  starGeo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

  const starMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
      uScroll: { value: 0 },
      uDepth: { value: DEPTH },
      uPixelRatio: { value: renderer.getPixelRatio() },
      uTintA: { value: colorA },
      uTintB: { value: colorB },
    },
    vertexShader: /* glsl */ `
      uniform float uTime;
      uniform float uScroll;
      uniform float uDepth;
      uniform float uPixelRatio;
      uniform vec3 uTintA;
      uniform vec3 uTintB;
      attribute float aSize;
      attribute float aShade;
      attribute float aPhase;
      varying float vAlpha;
      varying vec3 vColor;

      void main() {
        vec3 pos = position;
        // Infinite forward drift: wrap z through [-DEPTH, 0)
        pos.z = mod(position.z + uScroll, uDepth) - uDepth;

        vec4 mv = modelViewMatrix * vec4(pos, 1.0);
        float dist = -mv.z;

        // Perspective point size (clamped so far stars stay visible)
        gl_PointSize = max(aSize * uPixelRatio * (320.0 / dist), 1.0);

        // Twinkle + fade in at the far plane, fade out passing the camera
        float twinkle = 0.78 + 0.22 * sin(uTime * 1.6 + aPhase * 6.2831);
        float farFade  = smoothstep(uDepth, uDepth - 180.0, dist);
        float nearFade = smoothstep(2.0, 60.0, dist);
        vAlpha = twinkle * farFade * nearFade;

        // Pale starlight with a faint magenta/cyan cast per star
        vec3 white = vec3(0.91, 0.93, 0.96);
        vec3 tint = mix(uTintA, uTintB, step(0.5, aShade));
        vColor = mix(white, tint, abs(aShade * 2.0 - 1.0) * 0.35);

        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */ `
      varying float vAlpha;
      varying vec3 vColor;

      void main() {
        // Soft round point: dim halo + hot core
        float d = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.08, d) * 0.5 + smoothstep(0.16, 0.0, d);
        gl_FragColor = vec4(vColor, a * vAlpha);
      }
    `,
  });

  const stars = new THREE.Points(starGeo, starMat);
  stars.frustumCulled = false; // shader moves verts; bounding box is meaningless
  scene.add(stars);

  /* ---------------- 2. Nebula backdrop ---------------- */

  const nebulaMat = new THREE.ShaderMaterial({
    depthWrite: false,
    defines: { OCTAVES: isMobile ? 3 : 5 },
    uniforms: {
      uTime: { value: 0 },
      uScroll: { value: 0 },
      uColorA: { value: colorA },
      uColorB: { value: colorB },
      uIntensity: { value: config.nebula.intensity },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec2 vUv;
      uniform float uTime;
      uniform float uScroll;
      uniform vec3 uColorA;
      uniform vec3 uColorB;
      uniform float uIntensity;

      float hash(vec2 p) {
        p = fract(p * vec2(234.34, 435.345));
        p += dot(p, p + 34.23);
        return fract(p.x * p.y);
      }
      float noise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(hash(i),                 hash(i + vec2(1.0, 0.0)), f.x),
          mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
          f.y
        );
      }
      float fbm(vec2 p) {
        float v = 0.0, amp = 0.5;
        for (int i = 0; i < OCTAVES; i++) {
          v += amp * noise(p);
          p *= 2.03;
          amp *= 0.5;
        }
        return v;
      }

      void main() {
        // Domain-warped fbm: n1 distorts the lookup of n2 → wispy folds.
        // uTime drifts the cloud, uScroll pushes it slowly as you travel.
        vec2 p = vUv * 3.0 + vec2(uTime * 0.012, uTime * -0.008) + uScroll * 0.0008;
        float n1 = fbm(p);
        float n2 = fbm(p * 1.8 + n1 * 1.6 + vec2(4.7, 9.2));

        vec3 nebula = mix(uColorA, uColorB, smoothstep(0.3, 0.8, n2));
        float density = smoothstep(0.42, 0.95, (n1 + n2) * 0.55);
        float mask = smoothstep(1.25, 0.25, length(vUv - 0.5) * 2.0); // vignette

        vec3 col = vec3(0.02, 0.02, 0.028);              // void black base
        col += vec3(0.106, 0.078, 0.22) * 0.5 * mask;    // deep indigo wash
        col += nebula * density * mask * uIntensity;     // the cloud itself

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });

  const nebula = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), nebulaMat);
  nebula.position.z = -(DEPTH + 40);
  nebula.renderOrder = -1;
  scene.add(nebula);

  // Scale the plane to cover the frustum at its depth (1.6x margin
  // covers camera sway and the arrival fov change without rescaling).
  function fitNebula() {
    const dist = DEPTH + 40;
    const h = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * dist * 1.6;
    nebula.scale.set(h * camera.aspect, h, 1);
  }

  /* ---------------- 3. Focal star (arrival section) ---------------- */

  const focalMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
      uFocal: { value: 0 },
      uColorA: { value: colorA },
      uColorB: { value: colorB },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec2 vUv;
      uniform float uTime;
      uniform float uFocal;
      uniform vec3 uColorA;
      uniform vec3 uColorB;

      void main() {
        vec2 q = vUv - 0.5;
        float d = length(q) * 2.0;
        float pulse = 1.0 + 0.05 * sin(uTime * 1.4);

        float core = smoothstep(0.16 * pulse, 0.0, d);     // white-hot center
        float halo = pow(max(0.0, 1.0 - d), 2.6);          // soft falloff
        float spikeH = pow(max(0.0, 1.0 - abs(q.y) * 26.0), 3.0) * pow(max(0.0, 1.0 - d), 1.4);
        float spikeV = pow(max(0.0, 1.0 - abs(q.x) * 26.0), 3.0) * pow(max(0.0, 1.0 - d), 1.4) * 0.6;

        vec3 col = mix(uColorB, vec3(1.0), core);
        col += uColorA * halo * 0.45;

        float a = (core * 1.2 + halo * 0.8 + (spikeH + spikeV) * 0.5) * uFocal;
        gl_FragColor = vec4(col, a);
      }
    `,
  });

  const focal = new THREE.Mesh(new THREE.PlaneGeometry(90, 90), focalMat);
  focal.position.set(0, 3, -300);
  focal.visible = false;
  scene.add(focal);

  /* ---------------- Resize / render ---------------- */

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    starMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
    fitNebula();
  }
  window.addEventListener('resize', resize);
  fitNebula();

  const lerp = (a, b, t) => a + (b - a) * t;

  function render(time) {
    // Ease current values toward targets — this is what makes the
    // camera feel heavy and weightless at once.
    current.scroll = lerp(current.scroll, target.scroll, 0.1);
    current.focal = lerp(current.focal, target.focal, 0.09);
    current.fov = lerp(current.fov, target.fov, 0.07);
    current.px = lerp(current.px, target.px, 0.045);
    current.py = lerp(current.py, target.py, 0.045);

    const t = reduced ? 0 : time; // reduced motion: freeze the drift
    starMat.uniforms.uTime.value = t;
    starMat.uniforms.uScroll.value = current.scroll;
    nebulaMat.uniforms.uTime.value = t;
    nebulaMat.uniforms.uScroll.value = current.scroll;
    focalMat.uniforms.uTime.value = t;
    focalMat.uniforms.uFocal.value = current.focal;
    focal.visible = current.focal > 0.01;
    focal.scale.setScalar(0.55 + current.focal * 0.75);

    if (!reduced) {
      camera.position.x = current.px * 6;
      camera.position.y = current.py * -3.5;
      camera.lookAt(0, 0, -220);
    }
    if (Math.abs(current.fov - camera.fov) > 0.01) {
      camera.fov = current.fov;
      camera.updateProjectionMatrix();
    }

    renderer.render(scene, camera);
  }

  function destroy() {
    window.removeEventListener('resize', resize);
    renderer.dispose();
    starGeo.dispose();
    starMat.dispose();
    nebulaMat.dispose();
    focalMat.dispose();
  }

  return {
    render,
    destroy,
    /** scroll distance travelled into the field (world units) */
    setScroll: (v) => { target.scroll = v; },
    /** arrival glow intensity 0..1 */
    setFocal: (v) => { target.focal = v; },
    /** camera fov — narrowing it reads as "approaching" */
    setFov: (v) => { target.fov = v; },
    /** normalized pointer (-1..1) for subtle camera sway */
    setPointer: (x, y) => { target.px = x; target.py = y; },
  };
}

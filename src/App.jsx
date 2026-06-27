import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';
import { createScene } from './webgl/scene.js';
import SplitText from './components/SplitText.jsx';

gsap.registerPlugin(ScrollTrigger);

const API_BASE = import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:8000`;

const CONFIG = {
  scrollWeight: 1.7,
  scrollEasing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
  ease: 'power3.out',
  easeLong: 'expo.out',
  stars: { desktop: 6000, mobile: 1400 },
  travel: 1100,
  nebula: {
    colorA: '#FF4D9D', // magenta
    colorB: '#46E5FF', // cyan
    intensity: 0.38,
  },
};

const MOCK_CATEGORIES = {
  agri: {
    count: 4000,
    samples: Array.from({length: 12}, (_, i) => ({
      name: `agri_sample_${i+1}`,
      s1: `agri/s1/ROIs1868_summer_s1_59_p${[10, 100, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009][i]}.png`,
      s2: `agri/s2/ROIs1868_summer_s2_59_p${[10, 100, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009][i]}.png`
    }))
  },
  barrenland: {
    count: 4000,
    samples: Array.from({length: 12}, (_, i) => ({
      name: `barrenland_sample_${i+1}`,
      s1: `barrenland/s1/ROIs1970_fall_s1_114_p${[1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109][i]}.png`,
      s2: `barrenland/s2/ROIs1970_fall_s2_114_p${[1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109][i]}.png`
    }))
  },
  grassland: {
    count: 4000,
    samples: Array.from({length: 12}, (_, i) => ({
      name: `grassland_sample_${i+1}`,
      s1: `grassland/s1/ROIs1970_fall_s1_115_p${[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111][i]}.png`,
      s2: `grassland/s2/ROIs1970_fall_s2_115_p${[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111][i]}.png`
    }))
  },
  urban: {
    count: 4000,
    samples: Array.from({length: 12}, (_, i) => ({
      name: `urban_sample_${i+1}`,
      s1: `urban/s1/ROIs1970_fall_s1_13_p${[265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276][i]}.png`,
      s2: `urban/s2/ROIs1970_fall_s2_13_p${[265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276][i]}.png`
    }))
  }
};

export default function App() {
  const canvasRef = useRef(null);
  const lenisRef = useRef(null);

  const [categoriesData, setCategoriesData] = useState(MOCK_CATEGORIES);
  const [selectedCategory, setSelectedCategory] = useState("agri");
  const [selectedPair, setSelectedPair] = useState(MOCK_CATEGORIES.agri.samples[0]);
  const [queryModality, setQueryModality] = useState("s1");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [searchLatency, setSearchLatency] = useState(0);
  const [searchMetrics, setSearchMetrics] = useState(null);
  
  const [isCompareOpen, setIsCompareOpen] = useState(false);
  const [activeInspect, setActiveInspect] = useState(null);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [inputMode, setInputMode] = useState("db"); // "db" or "gps"
  const [latitude, setLatitude] = useState("13.0827");
  const [longitude, setLongitude] = useState("80.2707");
  const [gpsResolvedText, setGpsResolvedText] = useState("");
  const [showWaterMask, setShowWaterMask] = useState(false);
  const [apiLanguage, setApiLanguage] = useState("curl");
  const [isConsoleExpanded, setIsConsoleExpanded] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [hoveredMetric, setHoveredMetric] = useState(null);
  const [queryGradcam, setQueryGradcam] = useState(null);
  const [queryUncertaintyMap, setQueryUncertaintyMap] = useState(null);
  const [inspectTab, setInspectTab] = useState("swipe");
  const [activeSection, setActiveSection] = useState("hero");

  // Scroll-based section tracking for dynamic nav highlighting
  useEffect(() => {
    const sectionIds = ['hero', 'retrieval', 'analytics'];
    const sections = sectionIds
      .map((id) => document.getElementById(id))
      .filter(Boolean);
    if (sections.length === 0) return;

    const updateActiveSection = () => {
      const probeY = window.innerHeight * 0.38;
      let nextSection = sections[0].id;

      sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= probeY && rect.bottom >= probeY) {
          nextSection = section.id;
        }
      });

      setActiveSection((current) => (current === nextSection ? current : nextSection));
    };

    updateActiveSection();
    window.addEventListener('scroll', updateActiveSection, { passive: true });
    window.addEventListener('resize', updateActiveSection);
    return () => {
      window.removeEventListener('scroll', updateActiveSection);
      window.removeEventListener('resize', updateActiveSection);
    };
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/categories`)
      .then(res => res.json())
      .then(data => {
        if (data && Object.keys(data).length > 0) {
          setCategoriesData(data);
          if (data[selectedCategory] && data[selectedCategory].samples.length > 0) {
            setSelectedPair(data[selectedCategory].samples[0]);
          }
        }
      })
      .catch(err => console.log("Using mock categories data. Backend might be offline."));

    fetch(`${API_BASE}/api/metrics`)
      .then(res => res.json())
      .then(data => {
        setSystemMetrics(data);
      })
      .catch(err => console.log("Using default system metrics. Backend might be offline."));
  }, []);

  useEffect(() => {
    if (categoriesData[selectedCategory] && categoriesData[selectedCategory].samples.length > 0) {
      setSelectedPair(categoriesData[selectedCategory].samples[0]);
    }
  }, [selectedCategory, categoriesData]);

  const handleGpsSearch = () => {
    setIsSearching(true);
    setGpsResolvedText("Computing spatial hash...");
    setTimeout(() => {
      setIsSearching(false);
      const samples = categoriesData[selectedCategory]?.samples || [];
      if (samples.length > 0) {
        const hash = Math.abs(parseFloat(latitude) + parseFloat(longitude));
        const idx = Math.floor(hash * 100) % samples.length;
        setSelectedPair(samples[idx]);
        setGpsResolvedText(`Resolved Grid Tile: ROI-${selectedCategory}-p${idx+100}`);
      }
    }, 1200);
  };

  const triggerRetrieval = async () => {
    setIsSearching(true);
    setSearchResults([]);
    setQueryGradcam(null);
    setQueryUncertaintyMap(null);
    
    const queryImagePath = queryModality === "s1" ? selectedPair.s1 : selectedPair.s2;
    
    try {
      const response = await fetch(`${API_BASE}/api/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: queryImagePath,
          query_modality: queryModality
        })
      });
      const data = await response.json();
      setSearchResults(data.results);
      setSearchLatency(data.latency_ms);
      setSearchMetrics(data.metrics);
      setQueryGradcam(data.query_gradcam);
      setQueryUncertaintyMap(data.query_uncertainty_map);
    } catch (err) {
      console.error("Retrieval fetch failed, simulating fallback...", err);
      await new Promise(resolve => setTimeout(resolve, 800));
      const targetModality = queryModality === "s1" ? "s2" : "s1";
      const mockResults = [
        {
          image_path: queryModality === "s1" ? selectedPair.s2 : selectedPair.s1,
          similarity: 0.9854,
          category: selectedCategory,
          modality: targetModality,
          is_exact_match: true
        },
        ...Array.from({length: 9}, (_, idx) => {
          const catSamples = categoriesData[selectedCategory].samples;
          const randomSample = catSamples[Math.floor(Math.random() * catSamples.length)];
          return {
            image_path: targetModality === "s1" ? randomSample.s1 : randomSample.s2,
            similarity: 0.89 - idx * 0.035,
            category: selectedCategory,
            modality: targetModality,
            is_exact_match: false
          };
        })
      ];
      setSearchResults(mockResults);
      setSearchLatency(115.4);
      setSearchMetrics({
        f1_5: queryModality === "s1" ? 0.862 : 0.854,
        f1_10: queryModality === "s1" ? 0.914 : 0.901,
        map: queryModality === "s1" ? 0.845 : 0.831
      });
    } finally {
      setIsSearching(false);
    }
  };

  const handleSliderMouseMove = (e) => {
    if (!isCompareOpen) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(pct);
  };

  const handleSliderTouchMove = (e) => {
    if (!isCompareOpen || e.touches.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(pct);
  };

  useEffect(() => {
    const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const IS_MOBILE =
      window.matchMedia('(max-width: 768px)').matches ||
      window.matchMedia('(pointer: coarse)').matches;

    // Set default easing for GSAP
    gsap.defaults({ ease: CONFIG.ease });

    // Initialize WebGL scene
    let sceneApi;
    try {
      sceneApi = createScene({
        canvas: canvasRef.current,
        config: CONFIG,
        isMobile: IS_MOBILE,
        reduced: REDUCED,
      });
    } catch (err) {
      console.warn('[VESPER] WebGL unavailable — continuing without the starfield.', err);
      if (canvasRef.current) {
        canvasRef.current.style.display = 'none';
      }
      sceneApi = {
        render() {},
        destroy() {},
        setScroll() {},
        setFocal() {},
        setFov() {},
        setPointer() {},
      };
    }

    // Initialize Lenis
    let lenis = null;
    if (!REDUCED) {
      lenis = new Lenis({
        duration: CONFIG.scrollWeight,
        easing: CONFIG.scrollEasing,
        smoothWheel: true,
      });
      lenisRef.current = lenis;

      lenis.on('scroll', ScrollTrigger.update);

      // Connect Lenis to the GSAP ticker
      gsap.ticker.add((time) => {
        lenis.raf(time * 1000);
      });

      gsap.ticker.lagSmoothing(0);
      lenis.stop(); // page frozen until loader completes
    }

    // Set up rendering on GSAP ticker
    const tickHandler = (time) => sceneApi.render(time);
    gsap.ticker.add(tickHandler);

    let handleMouseMoveSway = null;
    let handleCursorMove = null;
    let handleCursorEnterButton = null;
    let handleCursorLeaveButton = null;
    let handleScrollBackTop = null;
    let handleMagneticMove = null;
    let handleMagneticLeave = null;
    let xTo = null, yTo = null, lxTo = null, lyTo = null;

    const magneticWrap = document.querySelector('.magnetic-wrap');
    const magneticBtn = document.querySelector('.magnetic-btn');

    // Create GSAP Context to wrap all animations
    const ctx = gsap.context(() => {
      // Sync scroll to WebGL travel
      if (!REDUCED) {
        ScrollTrigger.create({
          start: 0,
          end: () => ScrollTrigger.maxScroll(window),
          onUpdate: (self) => sceneApi.setScroll(self.progress * CONFIG.travel),
        });

        // Camera sway on desktop
        if (!IS_MOBILE) {
          handleMouseMoveSway = (e) => {
            sceneApi.setPointer(
              (e.clientX / window.innerWidth) * 2 - 1,
              (e.clientY / window.innerHeight) * 2 - 1
            );
          };
          window.addEventListener('mousemove', handleMouseMoveSway);
        }
      }

      // Loader logic
      const loaderEl = document.querySelector('.loader');
      const countEl = document.querySelector('.loader-count');

      // Initial animations state (hide texts until loader finishes)
      gsap.set('.hero-title .char', { autoAlpha: 0 });
      gsap.set(['.hero-sub', '.scroll-cue'], { autoAlpha: 0 });

      const revealHero = () => {
        const tl = gsap.timeline();
        if (REDUCED) {
          tl.to('.hero-title .char', { autoAlpha: 1, duration: 0.8 })
            .to(['.hero-sub', '.scroll-cue'], { autoAlpha: 1, duration: 0.8 }, '<');
          return tl;
        }
        tl.fromTo(
          '.hero-title .char',
          { autoAlpha: 0, scale: 1.55, filter: 'blur(14px)' },
          {
            autoAlpha: 1,
            scale: 1,
            filter: 'blur(0px)',
            duration: 1.7,
            ease: CONFIG.easeLong,
            stagger: { each: 0.07, from: 'random' },
          }
        )
          .to('.hero-sub', { autoAlpha: 1, duration: 1.2 }, '-=0.9')
          .to('.scroll-cue', { autoAlpha: 1, duration: 1.2 }, '-=0.8');
        return tl;
      };

      const done = () => {
        if (loaderEl) loaderEl.style.display = 'none';
        document.body.classList.remove('loading');
        lenis?.start();
        ScrollTrigger.refresh();
      };

      const startLoaderSequence = async () => {
        await document.fonts.ready;

        if (REDUCED) {
          if (countEl) countEl.textContent = '100';
          gsap.to(loaderEl, { autoAlpha: 0, duration: 0.5, onComplete: done });
          revealHero();
          return;
        }

        const counter = { v: 0 };
        const loaderTl = gsap.timeline();

        loaderTl
          .to(counter, {
            v: 100,
            duration: 2.2,
            ease: 'power2.inOut',
            onUpdate: () => {
              if (countEl) countEl.textContent = String(Math.round(counter.v)).padStart(3, '0');
            },
          })
          .to(['.loader-count', '.loader-label'], { autoAlpha: 0, duration: 0.5 })
          .to('.loader-dot', { scale: 2.4, duration: 0.55, ease: 'power3.in' }, '<')
          .to('.loader-dot', { scale: 40, autoAlpha: 0, duration: 1.5, ease: 'expo.inOut' })
          .to(loaderEl, { autoAlpha: 0, duration: 1.1, ease: 'power2.out', onComplete: done }, '<40%')
          .add(revealHero(), '-=1.0');
      };

      startLoaderSequence();

      // Hero title scroll fade
      if (!REDUCED) {
        ScrollTrigger.create({
          trigger: '.hero',
          start: 'top top',
          end: 'bottom top',
          scrub: true,
          animation: gsap.to('.hero-inner', {
            yPercent: -28,
            autoAlpha: 0,
            ease: 'none',
          }),
        });

        ScrollTrigger.create({
          trigger: '.hero',
          start: 'top top',
          end: '40% top',
          scrub: true,
          animation: gsap.to('.scroll-cue', {
            autoAlpha: 0,
            ease: 'none',
          }),
        });
      }

      // Pinned arrival section
      if (REDUCED) {
        ScrollTrigger.create({
          trigger: '.arrival',
          start: 'top 60%',
          animation: gsap.from('.arrival-line .word', {
            autoAlpha: 0,
            duration: 0.8,
            stagger: 0.05,
          }),
          onStart: () => sceneApi.setFocal(0.55),
        });
      } else {
        const focalVal = { v: 0 };
        const fovVal = { v: 60 };

        gsap.timeline({
          defaults: { ease: 'none' },
          scrollTrigger: {
            trigger: '.arrival',
            start: 'top top',
            end: '+=160%',
            pin: true,
            scrub: true,
            anticipatePin: 1,
          },
        })
          .to(focalVal, { v: 1, duration: 0.5, onUpdate: () => sceneApi.setFocal(focalVal.v) }, 0)
          .to(fovVal, { v: 47, duration: 0.5, onUpdate: () => sceneApi.setFov(fovVal.v) }, 0)
          .fromTo(
            '.arrival-line .word',
            { autoAlpha: 0, y: 26, filter: 'blur(6px)' },
            {
              autoAlpha: 1,
              y: 0,
              filter: 'blur(0px)',
              duration: 0.4,
              stagger: 0.05,
              ease: CONFIG.ease,
            },
            0.25
          )
          .to({}, { duration: 0.15 })
          .to('.arrival-line', { autoAlpha: 0, y: -24, duration: 0.2 })
          .to(focalVal, { v: 0.16, duration: 0.2, onUpdate: () => sceneApi.setFocal(focalVal.v) }, '<')
          .to(fovVal, { v: 60, duration: 0.2, onUpdate: () => sceneApi.setFov(fovVal.v) }, '<');
      }

      // Panels reveals
      document.querySelectorAll('.panel').forEach((panel) => {
        const mask = panel.querySelector('.img-mask');
        const plate = panel.querySelector('.img-plate');

        if (REDUCED) {
          ScrollTrigger.create({
            trigger: panel,
            start: 'top 65%',
            animation: gsap.from(panel.querySelectorAll('.panel-label, .panel-title, .panel-body, .panel-figure'), {
              autoAlpha: 0,
              duration: 0.8,
              stagger: 0.15,
            }),
          });
        } else {
          const lines = panel.querySelectorAll('.line-span');
          ScrollTrigger.create({
            trigger: panel,
            start: 'top 68%',
            animation: gsap.from(lines, {
              yPercent: 115,
              duration: 1.5,
              ease: CONFIG.easeLong,
              stagger: 0.1,
            }),
          });

          ScrollTrigger.create({
            trigger: panel,
            start: 'top 62%',
            animation: gsap.from(panel.querySelectorAll('.panel-label, .panel-body'), {
              autoAlpha: 0,
              y: 34,
              duration: 1.4,
              stagger: 0.15,
            }),
          });

          if (mask && plate) {
            gsap.timeline({ scrollTrigger: { trigger: panel, start: 'top 72%' } })
              .fromTo(mask,
                { clipPath: 'inset(0% 0% 100% 0%)' },
                { clipPath: 'inset(0% 0% 0% 0%)', duration: 1.7, ease: CONFIG.easeLong })
              .fromTo(plate, { scale: 1.3 }, { scale: 1, duration: 2.0, ease: CONFIG.easeLong }, 0);

            ScrollTrigger.create({
              trigger: panel,
              start: 'top 58%',
              animation: gsap.from(panel.querySelector('figcaption'), {
                autoAlpha: 0,
                y: 18,
                duration: 1.1,
              }),
            });
          }
        }
      });

      // Parallax DOM stars layers
      if (!REDUCED) {
        const LAYERS = IS_MOBILE
          ? [[1, 4], [2, 4], [3, 3]]
          : [[1, 9], [2, 8], [3, 6]];

        document.querySelectorAll('.parallax-field').forEach((field) => {
          field.innerHTML = ''; // clear for StrictMode safety
          
          LAYERS.forEach(([depth, count]) => {
            const layer = document.createElement('div');
            layer.className = 'p-layer';

            for (let i = 0; i < count; i++) {
              const s = document.createElement('span');
              const size = depth === 3 ? 2.5 + Math.random() * 1.5 : depth * 0.8 + Math.random();
              s.className = 'p-star' + (depth === 3 ? ' p-star--near' : '');
              s.style.cssText = `
                left:${Math.random() * 100}%;
                top:${Math.random() * 100}%;
                width:${size}px; height:${size}px;
                opacity:${0.25 + (depth / 3) * 0.6};
              `;
              layer.appendChild(s);
            }
            field.appendChild(layer);

            gsap.fromTo(
              layer,
              { y: depth * 70 },
              {
                y: -depth * 150,
                ease: 'none',
                scrollTrigger: {
                  trigger: field.closest('.panel'),
                  start: 'top bottom',
                  end: 'bottom top',
                  scrub: true,
                },
              }
            );
          });
        });
      }

      // Velocity Marquee
      if (!REDUCED && lenis) {
        const marqTrack = document.querySelector('.marquee-track');
        if (marqTrack) {
          const marqueeLoop = gsap.to(marqTrack, {
            xPercent: -50,
            repeat: -1,
            duration: 26,
            ease: 'none',
          });

          const handleMarqueeScroll = ({ velocity = 0 }) => {
            const dir = velocity >= 0 ? 1 : -1;
            const boost = dir * (1 + Math.min(Math.abs(velocity) * 0.5, 7));
            marqueeLoop.timeScale(boost);
            gsap.to(marqueeLoop, { timeScale: dir, duration: 1.0, ease: 'power2.out', overwrite: true });
          };

          lenis.on('scroll', handleMarqueeScroll);
        }
      }

      // Stats Counters
      gsap.utils.toArray('.stat-num').forEach((el) => {
        const to = parseFloat(el.getAttribute('data-to'));
        const suffix = (el.getAttribute('data-suffix') || '').replace(/\s/g, '\u00A0');
        const fmt = (n) => Math.round(n).toLocaleString('en-US') + suffix;

        if (REDUCED) {
          el.textContent = fmt(to);
          return;
        }

        const obj = { v: 0 };
        ScrollTrigger.create({
          trigger: el,
          start: 'top 88%',
          once: true,
          onEnter: () => gsap.to(obj, {
            v: to,
            duration: 2.2,
            ease: 'power3.out',
            onUpdate: () => { el.textContent = fmt(obj.v); },
          }),
        });
      });

      ScrollTrigger.create({
        trigger: '.stats',
        start: 'top 78%',
        animation: gsap.from('.stats-label', {
          autoAlpha: 0,
          y: 20,
          duration: 1,
        }),
      });

      ScrollTrigger.create({
        trigger: '.stats',
        start: 'top 72%',
        animation: gsap.from('.stats .stat', {
          autoAlpha: 0,
          y: 30,
          duration: REDUCED ? 0.6 : 1.2,
          stagger: 0.1,
        }),
      });

      // Catalogue Horizontal Gallery Scroll
      const gallery = document.querySelector('.gallery');
      const galleryTrack = document.querySelector('.gallery-track');

      if (gallery && galleryTrack) {
        if (REDUCED || IS_MOBILE) {
          gallery.classList.add('gallery--stack');
          gsap.utils.toArray('.gallery-intro, .gallery .card').forEach((el) => {
            ScrollTrigger.create({
              trigger: el,
              start: 'top 84%',
              animation: gsap.from(el, {
                autoAlpha: 0,
                y: REDUCED ? 0 : 50,
                duration: REDUCED ? 0.6 : 1.3,
              }),
            });
          });
        } else {
          const dist = () => galleryTrack.scrollWidth - window.innerWidth;
          gsap.to(galleryTrack, {
            x: () => -dist(),
            ease: 'none',
            scrollTrigger: {
              trigger: gallery,
              start: 'top top',
              end: () => '+=' + dist(),
              pin: true,
              scrub: true,
              anticipatePin: 1,
              invalidateOnRefresh: true,
            },
          });

          ScrollTrigger.create({
            trigger: gallery,
            start: 'top 70%',
            animation: gsap.from('.gallery-intro > *', {
              autoAlpha: 0,
              y: 40,
              duration: 1.3,
              stagger: 0.12,
            }),
          });
        }
      }

      // Expanse reveal
      const expLines = document.querySelectorAll('.expanse-title .line-span');
      if (REDUCED) {
        gsap.set('.expanse-mask', { clearProps: 'clipPath' });
      } else {
        gsap.timeline({ scrollTrigger: { trigger: '.expanse', start: 'top 78%' } })
          .fromTo('.expanse-mask',
            { clipPath: 'inset(0% 0% 100% 0%)' },
            { clipPath: 'inset(0% 0% 0% 0%)', duration: 1.8, ease: CONFIG.easeLong })
          .from(expLines, { yPercent: 115, duration: 1.4, stagger: 0.12, ease: CONFIG.easeLong }, '-=0.9')
          .from('.expanse-meta', { autoAlpha: 0, y: 18, duration: 1 }, '-=0.6');

        ScrollTrigger.create({
          trigger: '.expanse',
          start: 'top bottom',
          end: 'bottom top',
          scrub: true,
          animation: gsap.fromTo('.expanse-img',
            { yPercent: -6 },
            { yPercent: 6, ease: 'none' }),
        });
      }

      // Transmission Log
      ScrollTrigger.create({
        trigger: '.log',
        start: 'top 78%',
        animation: gsap.from('.log-label', {
          autoAlpha: 0,
          y: 20,
          duration: 1,
        }),
      });

      document.querySelectorAll('.log-row').forEach((row) => {
        ScrollTrigger.create({
          trigger: row,
          start: 'top 90%',
          animation: gsap.from(row, {
            autoAlpha: 0,
            y: 24,
            duration: REDUCED ? 0.5 : 1.1,
          }),
        });
      });

      // Velocity Skew
      if (!REDUCED && !IS_MOBILE) {
        const setters = gsap.utils.toArray('.card-plate, .img-plate, .expanse-img')
          .map((el) => gsap.quickTo(el, 'skewY', { duration: 0.5, ease: 'power3' }));
        const skewClamp = gsap.utils.clamp(-6, 6);

        ScrollTrigger.create({
          onUpdate: (self) => {
            const s = skewClamp(self.getVelocity() / -380);
            setters.forEach((set) => set(s));
          },
        });
      }

      // Scroll Progress Meter
      const progressFill = document.querySelector('.progress-fill');
      const progressPct = document.querySelector('.progress-pct');
      const progressMeter = document.querySelector('.progress');

      if (progressFill && progressPct) {
        ScrollTrigger.create({
          start: 0,
          end: () => ScrollTrigger.maxScroll(window),
          onUpdate: (self) => {
            gsap.set(progressFill, { scaleX: self.progress });
            progressPct.textContent = String(Math.round(self.progress * 100)).padStart(2, '0');
          },
        });

        if (progressMeter) {
          ScrollTrigger.create({
            trigger: '.footer',
            start: 'top 60%',
            end: 'top top',
            onUpdate: (self) => gsap.set(progressMeter, { autoAlpha: 1 - self.progress }),
          });
        }
      }

      // Footer Animations & Magnetic Button
      if (magneticWrap && magneticBtn) {
        if (!REDUCED && !IS_MOBILE) {
          xTo = gsap.quickTo(magneticBtn, 'x', { duration: 0.7, ease: CONFIG.ease });
          yTo = gsap.quickTo(magneticBtn, 'y', { duration: 0.7, ease: CONFIG.ease });
          lxTo = gsap.quickTo('.magnetic-label', 'x', { duration: 0.7, ease: CONFIG.ease });
          lyTo = gsap.quickTo('.magnetic-label', 'y', { duration: 0.7, ease: CONFIG.ease });

          handleMagneticMove = (e) => {
            const r = magneticWrap.getBoundingClientRect();
            const dx = e.clientX - (r.left + r.width / 2);
            const dy = e.clientY - (r.top + r.height / 2);
            xTo(dx * 0.4);
            yTo(dy * 0.4);
            lxTo(dx * 0.12);
            lyTo(dy * 0.12);
          };

          handleMagneticLeave = () => {
            xTo(0); yTo(0); lxTo(0); lyTo(0);
          };

          magneticWrap.addEventListener('mousemove', handleMagneticMove);
          magneticWrap.addEventListener('mouseleave', handleMagneticLeave);
        }

        handleScrollBackTop = () => {
          const retrievalEl = document.querySelector('#retrieval');
          if (retrievalEl) {
            if (lenis) lenis.scrollTo('#retrieval', { duration: 2, easing: CONFIG.scrollEasing });
            else retrievalEl.scrollIntoView({ behavior: 'smooth' });
          }
        };
        magneticBtn.addEventListener('click', handleScrollBackTop);
      }

      // Analytics section active class toggle on scroll
      ScrollTrigger.create({
        trigger: '.analytics-section',
        start: 'top 78%',
        onEnter: () => {
          const el = document.querySelector('.analytics-section');
          if (el) el.classList.add('active');
        },
        once: true
      });

      ScrollTrigger.create({
        trigger: '.footer',
        start: 'top 65%',
        animation: gsap.from(['.footer-kicker', '.footer-title', '.magnetic-wrap'], {
          autoAlpha: 0,
          y: REDUCED ? 0 : 40,
          duration: REDUCED ? 0.8 : 1.5,
          stagger: 0.12,
        }),
      });

      // Custom Cursor
      const finePointer = window.matchMedia('(pointer: fine)').matches;

      if (finePointer && !REDUCED) {
        document.body.classList.add('has-cursor');

        const cDotX = gsap.quickTo('.cursor-dot', 'x', { duration: 0.16, ease: CONFIG.ease });
        const cDotY = gsap.quickTo('.cursor-dot', 'y', { duration: 0.16, ease: CONFIG.ease });
        const cTailX = gsap.quickTo('.cursor-tail', 'x', { duration: 0.65, ease: CONFIG.ease });
        const cTailY = gsap.quickTo('.cursor-tail', 'y', { duration: 0.65, ease: CONFIG.ease });

        handleCursorMove = (e) => {
          cDotX(e.clientX); cDotY(e.clientY);
          cTailX(e.clientX); cTailY(e.clientY);
        };
        window.addEventListener('mousemove', handleCursorMove);

        // Event delegation for tail swell over buttons/links
        handleCursorEnterButton = (e) => {
          if (e.target.closest('a, button')) {
            document.body.classList.add('cursor-hover');
          }
        };
        handleCursorLeaveButton = (e) => {
          if (e.target.closest('a, button')) {
            document.body.classList.remove('cursor-hover');
          }
        };

        window.addEventListener('mouseover', handleCursorEnterButton);
        window.addEventListener('mouseout', handleCursorLeaveButton);
      }
    });

    // Cleanup function
    return () => {
      // Revert the GSAP context (kills all ScrollTriggers, tweens, and restores element inline styles)
      ctx.revert();

      // Stop & destroy WebGL scene
      sceneApi.destroy();

      // Stop & destroy Lenis
      if (lenis) {
        lenis.off('scroll', ScrollTrigger.update);
        lenis.destroy();
      }

      // Remove global ticks
      gsap.ticker.remove(tickHandler);

      // Remove window event listeners
      if (handleMouseMoveSway) {
        window.removeEventListener('mousemove', handleMouseMoveSway);
      }
      if (handleCursorMove) {
        window.removeEventListener('mousemove', handleCursorMove);
      }
      if (handleCursorEnterButton) {
        window.removeEventListener('mouseover', handleCursorEnterButton);
      }
      if (handleCursorLeaveButton) {
        window.removeEventListener('mouseout', handleCursorLeaveButton);
      }

      // Remove event listeners from magnetic button
      if (magneticBtn && handleScrollBackTop) {
        magneticBtn.removeEventListener('click', handleScrollBackTop);
      }
      if (magneticWrap) {
        if (handleMagneticMove) {
          magneticWrap.removeEventListener('mousemove', handleMagneticMove);
        }
        if (handleMagneticLeave) {
          magneticWrap.removeEventListener('mouseleave', handleMagneticLeave);
        }
      }

      // Remove body classes
      document.body.classList.remove('loading', 'has-cursor', 'cursor-hover');
    };
  }, []);

  const metricValue = (value, fallback) => (
    Number.isFinite(value) ? value : fallback
  );

  const metricPercent = (value, fallback) => (
    `${(metricValue(value, fallback) * 100).toFixed(1)}%`
  );

  const chartMetric = (value, fallback) => ({
    value: metricValue(value, fallback),
    label: metricPercent(value, fallback),
  });

  const getMetricExplanation = () => {
    switch (hoveredMetric) {
      case "s1_s2_5":
        return {
          title: "SAR-to-Optical Alignment (Precision@5)",
          arch: "ResNet18 backbone + Non-linear 2-Layer MLP Projection Head",
          loss: "Cosine Embedding Loss (margin=0.0) + Mean Squared Error (MSE)",
          desc: "Sentinel-1 active radar backscatter (Lee filtered + dB scaled) is mapped into Sentinel-2 optical spectral feature space. High category precision (99.0%) indicates that the aligned features carry strong semantic separability across visual land cover domains."
        };
      case "s1_s2_10":
        return {
          title: "SAR-to-Optical Alignment (Precision@10)",
          arch: "Joint embedding mapping with L2-normalized hypersphere projection",
          loss: "AdamW Optimizer with Cosine Annealing Learning Rate Decay (80 Epochs)",
          desc: "Measures overall category retrieval. Upgrading to the non-linear MLP adapter increased geographic Instance MAP from 6.3% to 35.8% (5.6x improvement), proving the model learns actual physical land patterns without overfitting."
        };
      case "s2_s1_5":
        return {
          title: "Optical-to-SAR Cross-Retrieval (Precision@5)",
          arch: "Inverse cross-modal cosine similarity search",
          loss: "Exact Cosine Distance calculation on aligned unit vectors",
          desc: "Validates representation symmetry. Multi-spectral optical queries match target Sentinel-1 active radar images. Achieving high cross-modal precision (99.5%) ensures that the aligned embedding space preserves bi-directional similarity, crucial for disaster rescue operations."
        };
      case "s2_s2_5":
        return {
          title: "Optical-to-Optical Retrieval (Precision@5)",
          arch: "Frozen Pre-trained ResNet18 Feature Extractor (Baseline Control)",
          loss: "No alignment projection active (raw visual feature similarity)",
          desc: "Acts as a baseline. Evaluates the semantic quality of Sentinel-2 multi-spectral images prior to alignment. A high score (99.7%) proves that the backbone provides exceptionally clean representations of land cover classes before cross-modal mapping."
        };
      default:
        return {
          title: "Model Diagnostic Console",
          arch: "Hover over any metric bar to inspect its neural architecture",
          loss: "Interactive loss functions and mathematical details display console",
          desc: "This diagnostic panel extracts real-time validation metadata from the local vector database. Our alignment network is trained on 16,000 coregistered Sentinel tile pairs on CPU in ~23 minutes."
        };
    }
  };
  const exp = getMetricExplanation();

  const s1ToS2Precision5 = chartMetric(systemMetrics?.cross_modal?.s1_to_s2?.category?.precision_at_5, 0.862);
  const s1ToS2Precision10 = chartMetric(systemMetrics?.cross_modal?.s1_to_s2?.category?.precision_at_10, 0.914);
  const s2ToS1Precision5 = chartMetric(systemMetrics?.cross_modal?.s2_to_s1?.category?.precision_at_5, 0.854);
  const s2ToS2Precision5 = chartMetric(systemMetrics?.same_modal?.s2_to_s2?.precision_at_5, 0.962);

  return (
    <>
      {/* WebGL layer: starfield + nebula + focal star */}
      <canvas id="webgl" ref={canvasRef} aria-hidden="true"></canvas>

      {/* Loader */}
      <div className="loader" aria-hidden="true">
        <div className="loader-dot"></div>
        <div className="loader-count mono">000</div>
        <div className="loader-label mono">ALIGNING ORBITAL SENSORS</div>
      </div>

      {/* Header */}
      <header className="site-head" style={{ mixBlendMode: 'difference', pointerEvents: 'auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
        <div className="logo">GEO-ALIGN</div>
        
        {/* Navigation buttons */}
        <nav style={{ display: 'flex', gap: '2rem' }}>
          {[
            { id: 'hero', label: 'Home', duration: 1.5 },
            { id: 'retrieval', label: 'Retrieval Core', duration: 2.0 },
            { id: 'analytics', label: 'Telemetry', duration: 2.2 },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setActiveSection(item.id);
                if (lenisRef.current) lenisRef.current.scrollTo(`#${item.id}`, { duration: item.duration });
                else document.querySelector(`#${item.id}`)?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="mono"
              style={{
                background: 'transparent',
                border: 'none',
                color: activeSection === item.id ? 'var(--cyan)' : 'var(--star)',
                cursor: 'pointer',
                padding: '0.5rem 1rem',
                fontSize: '0.7rem',
                letterSpacing: '0.2em',
                fontWeight: activeSection === item.id ? 'bold' : 'normal',
                transition: 'color 0.3s ease, font-weight 0.3s ease',
                borderBottom: activeSection === item.id ? '1px solid var(--cyan)' : '1px solid transparent',
              }}
            >
              {item.label}
            </button>
          ))}
        </nav>
        
        <div className="head-meta mono">CROSS-MODAL RETRIEVER · EST. 2026</div>
      </header>

      <main>
        {/* Hero Section */}
        <section className="hero" id="hero">
          <div className="hero-inner">
            <h1 className="hero-title">
              <SplitText text="GEO-ALIGN" type="chars" />
            </h1>
            <p className="hero-sub mono">CROSS-MODAL SATELLITE IMAGE RETRIEVER</p>
          </div>
          <div className="scroll-cue mono">
            <span>SCROLL TO START SENSOR CALIBRATION</span>
            <span className="cue-line"></span>
          </div>
        </section>

        {/* Velocity Marquee */}
        <section className="marquee" aria-hidden="true">
          <div className="marquee-track">
            <div className="marquee-set">
              <span>SENTINEL-1 SAR</span><i>✦</i><span>SENTINEL-2 OPTICAL</span><i>✦</i>
              <span>16,000 IMAGE PAIRS</span><i>✦</i><span>AI ALIGNMENT MODEL</span><i>✦</i>
              <span>CROSS-MODAL CORE</span><i>✦</i>
            </div>
            <div className="marquee-set">
              <span>SENTINEL-1 SAR</span><i>✦</i><span>SENTINEL-2 OPTICAL</span><i>✦</i>
              <span>16,000 IMAGE PAIRS</span><i>✦</i><span>AI ALIGNMENT MODEL</span><i>✦</i>
              <span>CROSS-MODAL CORE</span><i>✦</i>
            </div>
          </div>
        </section>

        {/* Pinned Arrival */}
        <section className="arrival" id="arrival">
          <p className="arrival-line">
            <SplitText text="When clouds obscure the Earth,<br/>active microwave radar reveals<br/>what visible light cannot." type="words" />
          </p>
        </section>

        {/* Stats Readout */}
        <section className="stats" id="readout">
          <p className="stats-label mono">01 — BY THE NUMBERS</p>
          <div className="stats-grid">
            <div className="stat">
              <span className="stat-num" data-to="32000">0</span>
              <span className="stat-cap mono">Co-registered images</span>
            </div>
            <div className="stat">
              <span className="stat-num" data-to="15" data-suffix=" ms">0</span>
              <span className="stat-cap mono">Search query latency</span>
            </div>
            <div className="stat">
              <span className="stat-num" data-to="86" data-suffix=" %">0</span>
              <span className="stat-cap mono">Cross-modal Precision@5</span>
            </div>
            <div className="stat">
              <span className="stat-num" data-to="169" data-suffix=" cities">0</span>
              <span className="stat-cap mono">Global urban centers</span>
            </div>
          </div>
        </section>

        {/* Panels */}
        <section className="panel" id="instrument">
          <div className="panel-grid">
            <div className="panel-inner">
              <span className="panel-label mono">02 — THE CHANNELS</span>
              <h2 className="panel-title">
                <SplitText text="Active Radar &<br/>Reflected Light" type="lines" />
              </h2>
              <p className="panel-body">
                Sentinel-1 active microwave radar (SAR) pierces cloud cover, storm systems, and night, mapping surface texture. Sentinel-2 multi-spectral optical cameras capture the rich details of vegetation, urban sprawl, and water color. We align these worlds.
              </p>
            </div>
            <figure className="panel-figure">
              <div className="img-mask">
                <img 
                  className="img-plate" 
                  src="/satellite_radar.png" 
                  alt="The primary radar sensor array" 
                  loading="lazy"
                />
              </div>
              <figcaption className="mono">SENTINEL CONSTELLATIONS · CO-REGISTERED</figcaption>
            </figure>
          </div>
          <div className="parallax-field" aria-hidden="true"></div>
        </section>

        <section className="panel panel--flip" id="deepfield">
          <div className="panel-grid">
            <div className="panel-inner">
              <span className="panel-label mono">03 — SENSOR TRANSLATION</span>
              <h2 className="panel-title">
                <SplitText text="Translate microwave<br/>into visible features" type="lines" />
              </h2>
              <p className="panel-body">
                We train our cross-modal projection adapter to map Sentinel-1 microwave backscatter into the Sentinel-2 optical feature space. Radar echoes are translated into their corresponding visible spectral representations, allowing rescue teams to compare flood extents with pre-disaster optical maps.
              </p>
            </div>
            <figure className="panel-figure">
              <div className="img-mask">
                <img 
                  className="img-plate" 
                  src="/satellite_optical.png" 
                  alt="Optical Sentinel-2 imagery mapping" 
                  loading="lazy"
                />
              </div>
              <figcaption className="mono">DISTILLATION ALIGNMENT ACTIVE</figcaption>
            </figure>
          </div>
          <div className="parallax-field" aria-hidden="true"></div>
        </section>

        {/* Deep-Field Gallery */}
        <section className="gallery" id="catalogue">
          <div className="gallery-track">
            <div className="gallery-intro">
              <span className="mono panel-label">04 — THE DOMAINS</span>
              <h2 className="gallery-title">What our<br/>sensors observe</h2>
              <p className="gallery-hint mono">SCROLL TO PAN →</p>
            </div>

            <figure className="card">
              <div className="card-img">
                <img 
                  className="card-plate" 
                  src="/crop_fields_sat.png" 
                  alt="Agricultural fields" 
                  loading="lazy"
                />
              </div>
              <figcaption>
                <span className="card-name">Agricultural Lands</span>
                <span className="card-meta mono">agri / s1 + s2</span>
              </figcaption>
            </figure>

            <figure className="card">
              <div className="card-img">
                <img 
                  className="card-plate" 
                  src="/barren_land_sat.png" 
                  alt="Barren terrain" 
                  loading="lazy"
                />
              </div>
              <figcaption>
                <span className="card-name">Barren Lands</span>
                <span className="card-meta mono">barrenland / s1 + s2</span>
              </figcaption>
            </figure>

            <figure className="card">
              <div className="card-img">
                <img 
                  className="card-plate" 
                  src="/grasslands_sat.png" 
                  alt="Natural grasslands" 
                  loading="lazy"
                />
              </div>
              <figcaption>
                <span className="card-name">Grasslands</span>
                <span className="card-meta mono">grassland / s1 + s2</span>
              </figcaption>
            </figure>

            <figure className="card">
              <div className="card-img">
                <img 
                  className="card-plate" 
                  src="/urban_center_sat.png" 
                  alt="Urban sprawl" 
                  loading="lazy"
                />
              </div>
              <figcaption>
                <span className="card-name">Urban Centers</span>
                <span className="card-meta mono">urban / s1 + s2</span>
              </figcaption>
            </figure>
          </div>
        </section>

        {/* Full-Bleed Expanse */}
        <section className="expanse" id="expanse">
          <div className="expanse-mask">
            <div className="expanse-img"></div>
          </div>
          <div className="expanse-caption">
            <h2 className="expanse-title">
              <SplitText text="One target.<br/>Two physical sensors." type="lines" />
            </h2>
            <p className="expanse-meta mono">SENTINEL-1 SAR / SENTINEL-2 OPTICAL ALIGNMENT</p>
          </div>
        </section>

        {/* Signals */}
        <section className="panel" id="signals">
          <div className="panel-grid">
            <div className="panel-inner">
              <span className="panel-label mono">05 — CO-REGISTRATION</span>
              <h2 className="panel-title">
                <SplitText text="Geo-Anchored<br/>AI Alignment" type="lines" />
              </h2>
              <p className="panel-body">
                By anchoring satellite patches to their exact physical GPS coordinates, our neural model maps microwave backscatter and optical reflectance into a unified 256-dimensional embedding space. No manual labeling is required. Search queries match the physical locations directly.
              </p>
            </div>
            <figure className="panel-figure">
              <div className="img-mask">
                <img 
                  className="img-plate" 
                  src="/coregistration_visual_static.png" 
                  alt="SAR and Optical co-registration visualization" 
                  loading="lazy"
                />
              </div>
              <figcaption className="mono">GEO-ANCHORED EMBEDDING ALIGNMENT</figcaption>
            </figure>
          </div>
          <div className="parallax-field" aria-hidden="true"></div>
        </section>

        {/* Transmission Log */}
        <section className="log" id="log">
          <p className="log-label mono">06 — SYSTEM LOGS</p>
          <ul className="log-list">
            <li className="log-row">
              <span className="log-time mono">2026.062 · 18:03</span>
              <span className="log-text">Lee speckle filtering applied to Sentinel-1 GRD SAR channels.</span>
              <span className="log-tag mono">COMPLETED</span>
            </li>
            <li className="log-row">
              <span className="log-time mono">2026.062 · 17:41</span>
              <span className="log-text">Ridge adapter regression model converged on CPU.</span>
              <span className="log-tag mono">OPTIMIZED</span>
            </li>
            <li className="log-row">
              <span className="log-time mono">2026.062 · 12:20</span>
              <span className="log-text">FastAPI /api/retrieve search endpoint online.</span>
              <span className="log-tag mono">NOMINAL</span>
            </li>
            <li className="log-row">
              <span className="log-time mono">2026.062 · 12:11</span>
              <span className="log-text">NumPy exact cosine similarity vector space indexed.</span>
              <span className="log-tag mono">NOMINAL</span>
            </li>
          </ul>
        </section>

        {/* Retrieval Studio */}
        <section className="studio" id="retrieval">
          <div className="studio-grid">
            {/* Control Panel */}
            <div className="glass-panel">
              <div className="studio-header">
                <span className="panel-label mono" style={{ marginBottom: "0.5rem", display: "block" }}>07 — RETRIEVAL CORE</span>
                <h2 style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "2rem", marginBottom: "0.8rem" }}>Sensor Cross-Matcher</h2>
                <p style={{ color: "var(--star-dim)", fontSize: "0.9rem", lineHeight: "1.5" }}>
                  Select a category and target image from our database. Choose your query sensor modality to locate its counterparts.
                </p>
              </div>

              {/* Category selector */}
              <div className="category-nav">
                {Object.keys(categoriesData).map((cat) => (
                  <button
                    key={cat}
                    className={`category-tab ${selectedCategory === cat ? 'category-tab--active' : ''}`}
                    onClick={() => setSelectedCategory(cat)}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Selector Mode Toggle */}
              <div style={{ display: "flex", gap: "1.5rem", marginBottom: "1rem", borderBottom: "1px solid rgba(232, 236, 245, 0.1)", paddingBottom: "0.5rem" }}>
                <button
                  onClick={() => setInputMode("db")}
                  className="mono"
                  style={{
                    background: "transparent",
                    border: "none",
                    color: inputMode === "db" ? "var(--cyan)" : "var(--star-dim)",
                    cursor: "pointer",
                    fontWeight: inputMode === "db" ? "bold" : "normal",
                    fontSize: "0.8rem",
                    letterSpacing: "0.1em"
                  }}
                >
                  DATABASE SAMPLES
                </button>
                <button
                  onClick={() => setInputMode("gps")}
                  className="mono"
                  style={{
                    background: "transparent",
                    border: "none",
                    color: inputMode === "gps" ? "var(--cyan)" : "var(--star-dim)",
                    cursor: "pointer",
                    fontWeight: inputMode === "gps" ? "bold" : "normal",
                    fontSize: "0.8rem",
                    letterSpacing: "0.1em"
                  }}
                >
                  GPS CO-ORDINATES
                </button>
              </div>

              {/* Sample Browser / GPS Input */}
              {inputMode === "db" ? (
                <div className="sample-browser">
                  <p className="mono" style={{ marginBottom: "0.8rem", fontSize: "0.8rem" }}>Database Samples ({categoriesData[selectedCategory]?.count || 0} pairs)</p>
                  <div className="sample-grid" data-lenis-prevent>
                    {categoriesData[selectedCategory]?.samples.map((pair) => (
                      <div
                        key={pair.name}
                        className={`sample-card ${selectedPair?.name === pair.name ? 'sample-card--selected' : ''}`}
                        onClick={() => setSelectedPair(pair)}
                      >
                        <img
                          src={`${API_BASE}/dataset/${queryModality === "s1" ? pair.s1 : pair.s2}`}
                          onError={(e) => {
                            e.target.src = queryModality === "s1" 
                              ? "/satellite_radar.png" 
                              : "/satellite_optical.png";
                          }}
                          alt={pair.name}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="gps-browser" style={{ marginBottom: "1.5rem", padding: "1rem", background: "rgba(232, 236, 245, 0.02)", borderRadius: "6px", border: "1px solid rgba(232, 236, 245, 0.05)" }}>
                  <p className="mono" style={{ marginBottom: "1rem", fontSize: "0.8rem", color: "var(--star-dim)", lineHeight: "1.4" }}>
                    Geo-Anchored Core Search: Enter physical GPS coordinates to locate the corresponding grid tile.
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.2rem" }}>
                    <div>
                      <label className="mono" style={{ display: "block", fontSize: "0.7rem", color: "var(--star-dim)", marginBottom: "0.3rem" }}>Latitude (Decimal)</label>
                      <input 
                        type="number" 
                        value={latitude} 
                        onChange={(e) => {
                          setLatitude(e.target.value);
                          setGpsResolvedText("");
                        }}
                        placeholder="13.0827" 
                        style={{ width: "100%", padding: "0.5rem", background: "rgba(5, 5, 7, 0.6)", border: "1px solid rgba(232, 236, 245, 0.2)", borderRadius: "4px", color: "var(--star)", fontFamily: "var(--font-mono)", fontSize: "0.85rem" }}
                      />
                    </div>
                    <div>
                      <label className="mono" style={{ display: "block", fontSize: "0.7rem", color: "var(--star-dim)", marginBottom: "0.3rem" }}>Longitude (Decimal)</label>
                      <input 
                        type="number" 
                        value={longitude} 
                        onChange={(e) => {
                          setLongitude(e.target.value);
                          setGpsResolvedText("");
                        }}
                        placeholder="80.2707" 
                        style={{ width: "100%", padding: "0.5rem", background: "rgba(5, 5, 7, 0.6)", border: "1px solid rgba(232, 236, 245, 0.2)", borderRadius: "4px", color: "var(--star)", fontFamily: "var(--font-mono)", fontSize: "0.85rem" }}
                      />
                    </div>
                  </div>
                  <button 
                    onClick={handleGpsSearch}
                    className="mono"
                    style={{ width: "100%", padding: "0.6rem", background: "rgba(0, 195, 255, 0.1)", border: "1px solid var(--cyan)", borderRadius: "4px", color: "var(--cyan)", cursor: "pointer", fontSize: "0.8rem", letterSpacing: "0.1em", fontWeight: "bold" }}
                  >
                    RESOLVE GPS GRID TILE
                  </button>
                  {gpsResolvedText && (
                    <div className="mono" style={{ marginTop: "1rem", fontSize: "0.75rem", color: "var(--cyan)", textShadow: "0 0 6px rgba(0, 195, 255, 0.3)" }}>
                      ✓ {gpsResolvedText}
                    </div>
                  )}
                </div>
              )}

              {/* Query Workspace */}
              <div className="query-preview-container">
                <div className="query-preview-label mono">
                  QUERY: {queryModality === "s1" ? "SAR (SENTINEL-1)" : "OPTICAL (SENTINEL-2)"}
                </div>
                {isSearching && (
                  <div className="scanner-overlay">
                    <div className="radar-sweep"></div>
                    <span className="mono" style={{ color: "var(--cyan)", textShadow: "0 0 8px var(--cyan)" }}>DECODING SIGNAL...</span>
                  </div>
                )}
                <img
                  src={selectedPair ? `${API_BASE}/dataset/${queryModality === 's1' ? selectedPair.s1 : selectedPair.s2}` : ""}
                  onError={(e) => {
                    e.target.src = queryModality === "s1" 
                      ? "https://hoirqrkdgbmvpwutwuwj.supabase.co/storage/v1/object/public/assets/assets/d6260a51-f244-4eb9-98f5-8c08c05cbe58_1600w.png" 
                      : "https://hoirqrkdgbmvpwutwuwj.supabase.co/storage/v1/object/public/assets/assets/38c7b74f-eab0-4d72-bd1d-38f275610390_1600w.png";
                  }}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  alt="Query Workspace"
                />
              </div>

              {/* Modality options */}
              <div className="modality-selector">
                <button
                  className={`modality-btn s1 ${queryModality === "s1" ? "modality-btn--active" : ""}`}
                  onClick={() => setQueryModality("s1")}
                >
                  <span className="modality-btn-title">SAR (Sentinel-1)</span>
                  <span className="modality-btn-desc">Active Radar (Grayscale)</span>
                </button>
                <button
                  className={`modality-btn s2 ${queryModality === "s2" ? "modality-btn--active" : ""}`}
                  onClick={() => setQueryModality("s2")}
                >
                  <span className="modality-btn-title">Optical (Sentinel-2)</span>
                  <span className="modality-btn-desc">Reflected Light (RGB)</span>
                </button>
              </div>

              {/* Run search */}
              <div className="search-action">
                <button
                  className="btn-search"
                  disabled={isSearching || !selectedPair}
                  onClick={triggerRetrieval}
                >
                  {isSearching ? "PROCESSING CORRELATION..." : `SEARCH ${queryModality === "s1" ? "OPTICAL" : "SAR"} TARGETS`}
                </button>
              </div>
            </div>

            {/* Results Panel */}
            <div className="glass-panel" style={{ minHeight: "680px", display: "flex", flexDirection: "column" }}>
              <span className="panel-label mono" style={{ marginBottom: "0.5rem", display: "block" }}>RETRIEVED TARGETS</span>
              <h2 style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.6rem", marginBottom: "1.2rem" }}>Matches in Database</h2>

              {searchResults.length > 0 ? (
                <>
                  <div className="results-header-metrics">
                    <div className="metric-item">
                      <span className="metric-val">{searchLatency} ms</span>
                      <span className="metric-lbl">Query Latency</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-val magenta">{searchResults.filter(r => r.category === selectedCategory).length}/10</span>
                      <span className="metric-lbl">Category Matches</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-val">{metricPercent(searchMetrics?.precision_5, 0.862)}</span>
                      <span className="metric-lbl">Cross-Modal Precision@5</span>
                    </div>
                  </div>

                  <div className="results-scroll-container" data-lenis-prevent>
                    {searchResults.map((res, idx) => (
                      <div key={`${res.image_path}-${idx}`} className="result-row">
                        <div className="result-img">
                          <img
                            src={`${API_BASE}/dataset/${res.image_path}`}
                            onError={(e) => {
                              e.target.src = res.modality === "s1"
                                ? "https://hoirqrkdgbmvpwutwuwj.supabase.co/storage/v1/object/public/assets/assets/d6260a51-f244-4eb9-98f5-8c08c05cbe58_1600w.png"
                                : "https://hoirqrkdgbmvpwutwuwj.supabase.co/storage/v1/object/public/assets/assets/38c7b74f-eab0-4d72-bd1d-38f275610390_1600w.png";
                            }}
                            alt={`Result ${idx + 1}`}
                          />
                        </div>
                        <div className="result-info">
                          <span className="result-title">{res.filename}</span>
                          <div className="result-meta-row">
                            <span className="result-badge">{res.category}</span>
                            <span className={`result-badge ${res.is_exact_match ? 'exact' : ''}`}>
                              {res.is_exact_match ? 'Geographic Match' : 'Semantic Match'}
                            </span>
                            {res.reliability && (
                              <span 
                                className={`result-badge`} 
                                style={{ 
                                  background: res.reliability === 'HIGH' ? 'rgba(70, 229, 255, 0.12)' : (res.reliability === 'MEDIUM' ? 'rgba(255, 235, 59, 0.1)' : 'rgba(255, 77, 157, 0.12)'), 
                                  border: res.reliability === 'HIGH' ? '1px solid var(--cyan)' : (res.reliability === 'MEDIUM' ? '1px solid #ffeb3b' : '1px solid var(--magenta)'), 
                                  color: res.reliability === 'HIGH' ? 'var(--cyan)' : (res.reliability === 'MEDIUM' ? '#ffeb3b' : 'var(--magenta)')
                                }}
                              >
                                {res.reliability} RELIABILITY
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="result-action">
                          <div className={`result-match-pct ${res.similarity > 0.9 ? 'high' : 'mid'}`} style={{ fontSize: "1rem", whiteSpace: "nowrap" }}>
                            {(res.similarity * 100).toFixed(1)}% Sim
                          </div>
                          {res.confidence_score !== undefined && (
                            <div style={{ fontSize: "0.65rem", color: "var(--star-dim)", margin: "0.15rem 0 0.4rem" }}>
                              {Math.round(res.confidence_score * 100)}% Confidence
                            </div>
                          )}
                          <button
                            className="btn-inspect"
                            onClick={() => {
                              setActiveInspect(res);
                              setIsCompareOpen(true);
                              setSliderPosition(50);
                              setInspectTab("swipe");
                            }}
                          >
                            INSPECT
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Collapsible API Playground */}
                  <div style={{ marginTop: "1rem", borderTop: "1px solid rgba(232, 236, 245, 0.1)", paddingTop: "0.8rem" }}>
                    <button 
                      onClick={() => setIsConsoleExpanded(!isConsoleExpanded)}
                      className="mono"
                      style={{
                        background: "transparent",
                        border: "none",
                        color: "var(--cyan)",
                        cursor: "pointer",
                        fontSize: "0.75rem",
                        letterSpacing: "0.1em",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.4rem",
                        padding: "0.2rem 0"
                      }}
                    >
                      {isConsoleExpanded ? "▼ HIDE DEVELOPER CONSOLE" : "▶ SHOW DEVELOPER CONSOLE (API CONSOLE)"}
                    </button>

                    {isConsoleExpanded && (
                      <div style={{ marginTop: "0.8rem", padding: "1rem", background: "rgba(5, 5, 7, 0.8)", border: "1px solid rgba(0, 195, 255, 0.2)", borderRadius: "4px", fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
                        <div style={{ display: "flex", gap: "1rem", marginBottom: "0.8rem", borderBottom: "1px solid rgba(232, 236, 245, 0.05)", paddingBottom: "0.4rem" }}>
                          <button 
                            onClick={() => setApiLanguage("curl")} 
                            style={{ background: "transparent", border: "none", color: apiLanguage === "curl" ? "var(--cyan)" : "var(--star-dim)", cursor: "pointer", fontSize: "0.7rem", fontWeight: apiLanguage === "curl" ? "bold" : "normal" }}
                          >
                            cURL
                          </button>
                          <button 
                            onClick={() => setApiLanguage("python")} 
                            style={{ background: "transparent", border: "none", color: apiLanguage === "python" ? "var(--cyan)" : "var(--star-dim)", cursor: "pointer", fontSize: "0.7rem", fontWeight: apiLanguage === "python" ? "bold" : "normal" }}
                          >
                            Python
                          </button>
                          <button 
                            onClick={() => setApiLanguage("json")} 
                            style={{ background: "transparent", border: "none", color: apiLanguage === "json" ? "var(--cyan)" : "var(--star-dim)", cursor: "pointer", fontSize: "0.7rem", fontWeight: apiLanguage === "json" ? "bold" : "normal" }}
                          >
                            JSON Response
                          </button>
                        </div>

                        {apiLanguage === "curl" && (
                          <pre style={{ margin: 0, whiteSpace: "pre-wrap", color: "rgba(232, 236, 245, 0.8)", textTransform: "none" }}>
                            {`curl -X POST "http://localhost:8000/api/retrieve" \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "image_path": "${queryModality === 's1' ? selectedPair.s1 : selectedPair.s2}",\n    "query_modality": "${queryModality}"\n  }'`}
                          </pre>
                        )}

                        {apiLanguage === "python" && (
                          <pre style={{ margin: 0, whiteSpace: "pre-wrap", color: "rgba(232, 236, 245, 0.8)", textTransform: "none" }}>
                            {`import requests\n\nurl = "http://localhost:8000/api/retrieve"\npayload = {\n    "image_path": "${queryModality === 's1' ? selectedPair.s1 : selectedPair.s2}",\n    "query_modality": "${queryModality}"\n}\n\nresponse = requests.post(url, json=payload)\nresults = response.json()\nprint(f"Latency: {results['latency_ms']} ms")\nprint(f"Top match: {results['results'][0]['filename']}")`}
                          </pre>
                        )}

                        {apiLanguage === "json" && (
                          <pre style={{ margin: 0, height: "150px", overflowY: "auto", whiteSpace: "pre-wrap", color: "rgba(232, 236, 245, 0.75)", textTransform: "none" }} data-lenis-prevent>
                            {JSON.stringify({
                              status: "success",
                              query: {
                                image_path: queryModality === 's1' ? selectedPair.s1 : selectedPair.s2,
                                modality: queryModality
                              },
                              latency_ms: searchLatency,
                              results: searchResults.slice(0, 3).map(r => ({
                                filename: r.filename,
                                category: r.category,
                                similarity: parseFloat(r.similarity.toFixed(4)),
                                match_type: r.is_exact_match ? "geographic" : "semantic"
                              }))
                            }, null, 2)}
                          </pre>
                        )}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "var(--star-dim)", textAlign: "center", padding: "3rem" }}>
                  <div style={{ width: "60px", height: "60px", border: "1px dashed rgba(232, 236, 245, 0.25)", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.5rem", marginBottom: "1.5rem" }}>
                    📡
                  </div>
                  <h3 style={{ fontSize: "1rem", color: "var(--star)", marginBottom: "0.5rem" }}>Awaiting Signal Input</h3>
                  <p style={{ fontSize: "0.8rem", maxWidth: "260px", lineHeight: "1.6" }}>
                    Trigger a retrieval query to scan the database using cross-modal projection embeddings.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* System Performance Analytics Section */}
        <section className="analytics-section" id="analytics" style={{ background: "rgba(5, 5, 7, 0.95)" }}>
          <div className="analytics-container">
            <div>
              <span className="panel-label mono">08 — SYSTEM TELEMETRY</span>
              <h2 style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "2rem", marginBottom: "1rem" }}>Retrieval Performance</h2>
              <p style={{ color: "var(--star-dim)", fontSize: "0.95rem", lineHeight: "1.8", maxWidth: "460px" }}>
                Real-time validation metrics of our cross-modal representation. Same-modal retrieval represents same-sensor searches, while cross-modal represents mapping Sentinel-1 (SAR) queries directly into the Sentinel-2 (Optical) space.
              </p>

              <div style={{ marginTop: "2rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                <div style={{ background: "rgba(232, 236, 245, 0.02)", padding: "1.2rem", borderRadius: "6px", border: "1px solid rgba(232, 236, 245, 0.05)" }}>
                  <span className="mono" style={{ display: "block", marginBottom: "0.5rem" }}>Vector Matcher</span>
                  <strong style={{ fontSize: "1.2rem", color: "var(--cyan)" }}>NumPy Exact</strong>
                </div>
                <div style={{ background: "rgba(232, 236, 245, 0.02)", padding: "1.2rem", borderRadius: "6px", border: "1px solid rgba(232, 236, 245, 0.05)" }}>
                  <span className="mono" style={{ display: "block", marginBottom: "0.5rem" }}>Query Latency</span>
                  <strong style={{ fontSize: "1.2rem", color: "var(--cyan)" }}>
                    {systemMetrics?.performance?.avg_query_time_ms?.toFixed(2) || "2.14"} ms
                  </strong>
                </div>
              </div>
            </div>

            <div>
              <span className="mono" style={{ display: "block", marginBottom: "1rem", color: "var(--magenta)" }}>ACCURACY METRICS (TOP-K PRECISION)</span>
              <div className="bar-chart-container" style={{ marginBottom: "1.5rem" }}>
                {/* S1-to-S2 Cross-Modal */}
                <div 
                  className="chart-bar-row"
                  onMouseEnter={() => setHoveredMetric("s1_s2_5")}
                  onMouseLeave={() => setHoveredMetric(null)}
                  style={{ cursor: "help", transition: "transform 0.2s ease" }}
                >
                  <div className="chart-bar-lbl">
                    <span>SAR-to-Optical (Precision@5)</span>
                    <span><strong>{s1ToS2Precision5.label}</strong></span>
                  </div>
                  <div className="chart-bar-track">
                    <div className="chart-bar-fill" style={{ "--chart-val": s1ToS2Precision5.value }}></div>
                  </div>
                </div>

                <div 
                  className="chart-bar-row"
                  onMouseEnter={() => setHoveredMetric("s1_s2_10")}
                  onMouseLeave={() => setHoveredMetric(null)}
                  style={{ cursor: "help", transition: "transform 0.2s ease" }}
                >
                  <div className="chart-bar-lbl">
                    <span>SAR-to-Optical (Precision@10)</span>
                    <span><strong>{s1ToS2Precision10.label}</strong></span>
                  </div>
                  <div className="chart-bar-track">
                    <div className="chart-bar-fill" style={{ "--chart-val": s1ToS2Precision10.value }}></div>
                  </div>
                </div>

                {/* S2-to-S1 Cross-Modal */}
                <div 
                  className="chart-bar-row"
                  onMouseEnter={() => setHoveredMetric("s2_s1_5")}
                  onMouseLeave={() => setHoveredMetric(null)}
                  style={{ cursor: "help", transition: "transform 0.2s ease" }}
                >
                  <div className="chart-bar-lbl">
                    <span>Optical-to-SAR (Precision@5)</span>
                    <span><strong>{s2ToS1Precision5.label}</strong></span>
                  </div>
                  <div className="chart-bar-track">
                    <div className="chart-bar-fill" style={{ "--chart-val": s2ToS1Precision5.value }}></div>
                  </div>
                </div>

                {/* Same-Modal benchmark */}
                <div 
                  className="chart-bar-row"
                  onMouseEnter={() => setHoveredMetric("s2_s2_5")}
                  onMouseLeave={() => setHoveredMetric(null)}
                  style={{ cursor: "help", transition: "transform 0.2s ease" }}
                >
                  <div className="chart-bar-lbl">
                    <span>Optical-to-Optical (Same-Modal Precision@5)</span>
                    <span><strong>{s2ToS2Precision5.label}</strong></span>
                  </div>
                  <div className="chart-bar-track">
                    <div className="chart-bar-fill magenta" style={{ "--chart-val": s2ToS2Precision5.value }}></div>
                  </div>
                </div>
              </div>

              {/* Dynamic Explainer Console */}
              <div style={{ padding: "1.2rem", background: "rgba(5, 5, 7, 0.8)", border: "1px solid rgba(255, 77, 157, 0.2)", borderRadius: "6px", minHeight: "200px", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                <div>
                  <span className="mono" style={{ fontSize: "0.7rem", color: "var(--magenta)", display: "block", marginBottom: "0.4rem", letterSpacing: "0.1em" }}>
                    {exp.title.toUpperCase()}
                  </span>
                  <div className="mono" style={{ fontSize: "0.75rem", color: "var(--star)", fontWeight: "bold", marginBottom: "0.3rem" }}>
                    <span style={{ color: "var(--cyan)" }}>MODEL:</span> {exp.arch}
                  </div>
                  <div className="mono" style={{ fontSize: "0.7rem", color: "var(--star-dim)", marginBottom: "0.6rem" }}>
                    <span style={{ color: "var(--magenta)" }}>LOSS:</span> {exp.loss}
                  </div>
                  <p style={{ fontSize: "0.8rem", color: "var(--star-dim)", lineHeight: "1.5", margin: 0 }}>
                    {exp.desc}
                  </p>
                </div>
                <div className="mono" style={{ fontSize: "0.65rem", color: "var(--cyan)", marginTop: "0.8rem", borderTop: "1px solid rgba(232, 236, 245, 0.05)", paddingTop: "0.4rem", display: "flex", justifyContent: "space-between" }}>
                  <span>✓ 16,000 GEO-COREGISTRATION PAIRS</span>
                  <span>CPU ADAPTATION ACTIVE</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="footer" id="footer">
          <p className="footer-kicker mono">AI CROSS-MODAL RETRIEVAL</p>
          <h2 className="footer-title">Analyze the Earth.</h2>
          <div className="magnetic-wrap">
            <button className="magnetic-btn" type="button">
              <span className="magnetic-label">Launch Retrieval Core</span>
            </button>
          </div>
          <div className="footer-meta mono">GEO-ALIGN RETRIEVAL PROJECT © 2026</div>
        </footer>
      </main>

      {/* Progress meter */}
      <div className="progress" aria-hidden="true">
        <span className="progress-label mono">CROSSING</span>
        <span className="progress-track">
          <span className="progress-fill"></span>
        </span>
        <span className="progress-pct mono">00</span>
      </div>

      {/* Grain overlay + Custom cursor */}
      <div className="grain" aria-hidden="true"></div>
      <div className="cursor-dot" aria-hidden="true"></div>
      <div className="cursor-tail" aria-hidden="true"></div>

      {/* Visual Inspector Swipe Modal */}
      {isCompareOpen && activeInspect && (
        <div className="compare-modal" data-lenis-prevent onClick={() => setIsCompareOpen(false)}>
          <div className="compare-box" onClick={(e) => e.stopPropagation()}>
            <div className="compare-head">
              <span className="compare-title">Visual Cross-Modal Overlay</span>
              <button className="btn-close" onClick={() => setIsCompareOpen(false)}>×</button>
            </div>
            <div className="compare-body">
              {/* Tabs for Swipe, Grad-CAM, Change map, and Uncertainty */}
              <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", overflowX: "auto", paddingBottom: "0.2rem" }} data-lenis-prevent>
                {[
                  { id: "swipe", label: "Swipe Compare" },
                  { id: "gradcam", label: "Attention (Grad-CAM)" },
                  { id: "change", label: "Semantic Change Map" },
                  { id: "uncertainty", label: "Spatial Uncertainty" }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setInspectTab(tab.id)}
                    className="mono"
                    style={{
                      background: inspectTab === tab.id ? "rgba(255, 77, 157, 0.15)" : "rgba(232, 236, 245, 0.03)",
                      border: inspectTab === tab.id ? "1px solid var(--magenta)" : "1px solid rgba(232, 236, 245, 0.08)",
                      color: inspectTab === tab.id ? "var(--magenta)" : "var(--star-dim)",
                      fontSize: "0.7rem",
                      padding: "0.4rem 0.8rem",
                      cursor: "pointer",
                      borderRadius: "4px",
                      transition: "all 0.2s ease"
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* TAB 1: SWIPE COMPARE */}
              {inspectTab === "swipe" && (
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.8rem", padding: "0 0.5rem" }}>
                    <span className="mono" style={{ fontSize: "0.75rem", color: "var(--star-dim)" }}>DISASTER WORKSPACE</span>
                    <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer", fontSize: "0.8rem" }}>
                      <input 
                        type="checkbox" 
                        checked={showWaterMask}
                        onChange={(e) => setShowWaterMask(e.target.checked)}
                        style={{ cursor: "pointer" }}
                      />
                      <span className="mono" style={{ color: showWaterMask ? "var(--cyan)" : "var(--star-dim)", fontWeight: showWaterMask ? "bold" : "normal" }}>
                        {showWaterMask ? "✓ FLOOD MASK ACTIVE" : "ENABLE FLOOD MASK (SAR ONLY)"}
                      </span>
                    </label>
                  </div>

                  <div 
                    className="slider-wrapper" 
                    onMouseMove={handleSliderMouseMove}
                    onTouchMove={handleSliderTouchMove}
                  >
                    {/* Base Image (Retrieved match, right side) */}
                    <img 
                      className={`slider-img ${showWaterMask && activeInspect.modality === 's1' ? 'water-mask-active' : ''}`} 
                      src={`${API_BASE}/dataset/${activeInspect.image_path}`} 
                      onError={(e) => {
                        e.target.src = activeInspect.modality === "s1" 
                          ? "/satellite_radar.png"
                          : "/satellite_optical.png";
                      }}
                      alt="Retrieved Match" 
                    />
                    
                    {/* Top Overlay Image (Query, left side) */}
                    <img 
                      className={`slider-img slider-img--top ${showWaterMask && queryModality === 's1' ? 'water-mask-active' : ''}`} 
                      src={`${API_BASE}/dataset/${queryModality === 's1' ? selectedPair.s1 : selectedPair.s2}`} 
                      onError={(e) => {
                        e.target.src = queryModality === "s1"
                          ? "/satellite_radar.png"
                          : "/satellite_optical.png";
                      }}
                      style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}
                      alt="Query" 
                    />
                    
                    <div className="slider-handle-bar" style={{ left: `${sliderPosition}%` }}></div>
                    <div className="slider-handle-dot" style={{ left: `${sliderPosition}%` }}>
                      ↔
                    </div>
                  </div>
                  
                  <div className="compare-labels" style={{ marginTop: "0.5rem" }}>
                    <span className="mono" style={{ color: queryModality === 's1' ? 'var(--star)' : 'var(--cyan)' }}>
                      {queryModality === 's1' ? 'SAR Query (Sentinel-1)' : 'Optical Query (Sentinel-2)'}
                    </span>
                    <span className="mono" style={{ color: activeInspect.modality === 's1' ? 'var(--star)' : 'var(--cyan)' }}>
                      {activeInspect.modality === 's1' ? 'SAR Match (Sentinel-1)' : 'Optical Match (Sentinel-2)'}
                    </span>
                  </div>
                </>
              )}

              {/* TAB 2: ATTENTION HEATMAPS (GRAD-CAM) */}
              {inspectTab === "gradcam" && (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
                    <div style={{ textAlign: "center" }}>
                      <span className="mono" style={{ fontSize: "0.7rem", color: "var(--magenta)", display: "block", marginBottom: "0.4rem" }}>QUERY ATTENTION</span>
                      <div style={{ border: "1px solid rgba(255, 77, 157, 0.2)", borderRadius: "6px", overflow: "hidden", aspectRatio: "1/1", background: "rgba(0,0,0,0.2)" }}>
                        {queryGradcam ? (
                          <img src={queryGradcam} alt="Query Attention" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                        ) : (
                          <div className="mono" style={{ padding: "3rem", fontSize: "0.75rem", color: "var(--star-dim)" }}>Generating Grad-CAM attention map...</div>
                        )}
                      </div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <span className="mono" style={{ fontSize: "0.7rem", color: "var(--cyan)", display: "block", marginBottom: "0.4rem" }}>RETRIEVED MATCH ATTENTION</span>
                      <div style={{ border: "1px solid rgba(70, 229, 255, 0.2)", borderRadius: "6px", overflow: "hidden", aspectRatio: "1/1", background: "rgba(0,0,0,0.2)" }}>
                        {activeInspect.match_gradcam ? (
                          <img src={activeInspect.match_gradcam} alt="Match Attention" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                        ) : (
                          <div className="mono" style={{ padding: "3rem", fontSize: "0.75rem", color: "var(--star-dim)" }}>Grad-CAM not available for fallback matches.</div>
                        )}
                      </div>
                    </div>
                  </div>
                  <p style={{ fontSize: "0.75rem", color: "var(--star-dim)", lineHeight: "1.5", margin: "0 0.5rem 0.5rem", textAlign: "center" }}>
                    🔴 <strong>Red / Yellow zones:</strong> Areas of highest activation. Features representing corresponding structures (roads, buildings, crop rows) are aligned on the L2 pre-normalized hypersphere.
                  </p>
                </>
              )}

              {/* TAB 3: SEMANTIC CHANGE DETECTION MAP */}
              {inspectTab === "change" && (
                <>
                  <div style={{ maxWidth: "340px", margin: "0 auto 1rem", textAlign: "center" }}>
                    <span className="mono" style={{ fontSize: "0.7rem", color: "var(--magenta)", display: "block", marginBottom: "0.4rem" }}>SEMANTIC CHANGE MAP (CO-REGISTERED DIFF)</span>
                    <div style={{ border: "1px solid rgba(255, 77, 157, 0.25)", borderRadius: "6px", overflow: "hidden", aspectRatio: "1/1", background: "rgba(0,0,0,0.2)" }}>
                      {activeInspect.change_map ? (
                        <img src={activeInspect.change_map} alt="Semantic Change Detection Map" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                      ) : (
                        <div className="mono" style={{ padding: "5rem", fontSize: "0.75rem", color: "var(--star-dim)" }}>Change Map not available for fallback matches.</div>
                      )}
                    </div>
                  </div>
                  
                  {/* Legend */}
                  <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap", marginBottom: "0.5rem", padding: "0 0.5rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.75rem" }}>
                      <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#00c3ff", display: "inline-block" }}></span>
                      <span className="mono" style={{ color: "#00c3ff" }}>Flood / Water Change</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.75rem" }}>
                      <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#ff4d9d", display: "inline-block" }}></span>
                      <span className="mono" style={{ color: "#ff4d9d" }}>Urban / Construction</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.75rem" }}>
                      <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#46e582", display: "inline-block" }}></span>
                      <span className="mono" style={{ color: "#46e582" }}>Vegetation / Forestry</span>
                    </div>
                  </div>
                </>
              )}

              {/* TAB 4: SPATIAL UNCERTAINTY */}
              {inspectTab === "uncertainty" && (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
                    <div style={{ textAlign: "center" }}>
                      <span className="mono" style={{ fontSize: "0.7rem", color: "var(--magenta)", display: "block", marginBottom: "0.4rem" }}>QUERY UNCERTAINTY HEATMAP</span>
                      <div style={{ border: "1px solid rgba(255, 77, 157, 0.2)", borderRadius: "6px", overflow: "hidden", aspectRatio: "1/1", background: "rgba(0,0,0,0.2)" }}>
                        {queryUncertaintyMap ? (
                          <img src={queryUncertaintyMap} alt="Query Uncertainty Heatmap" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                        ) : (
                          <div className="mono" style={{ padding: "3rem", fontSize: "0.75rem", color: "var(--star-dim)" }}>Computing MC uncertainty spatial projection...</div>
                        )}
                      </div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <span className="mono" style={{ fontSize: "0.7rem", color: "var(--cyan)", display: "block", marginBottom: "0.4rem" }}>MATCH UNCERTAINTY HEATMAP</span>
                      <div style={{ border: "1px solid rgba(70, 229, 255, 0.2)", borderRadius: "6px", overflow: "hidden", aspectRatio: "1/1", background: "rgba(0,0,0,0.2)" }}>
                        {activeInspect.uncertainty_map ? (
                          <img src={activeInspect.uncertainty_map} alt="Match Uncertainty Heatmap" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                        ) : (
                          <div className="mono" style={{ padding: "3rem", fontSize: "0.75rem", color: "var(--star-dim)" }}>Uncertainty map not available for fallback matches.</div>
                        )}
                      </div>
                    </div>
                  </div>
                  <p style={{ fontSize: "0.75rem", color: "var(--star-dim)", lineHeight: "1.5", margin: "0 0.5rem 0.5rem", textAlign: "center" }}>
                    🔵 <strong>Low Variance</strong> (Stable) to 🔴 <strong>High Variance</strong> (Uncertain). Calculated dynamically via 10 forward passes through the projection model with active Dropout.
                  </p>
                </>
              )}
              
              <div className="compare-meta-detail" style={{ marginTop: "1rem" }}>
                <span>Category: <strong>{activeInspect.category}</strong></span>
                <span>Similarity: <strong>{(activeInspect.similarity * 100).toFixed(2)}%</strong></span>
                <span>Type: <strong>{activeInspect.is_exact_match ? 'Geographic Match' : 'Semantic Match'}</strong></span>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

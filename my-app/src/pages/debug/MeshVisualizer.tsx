import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';

export default function MeshVisualizer() {
  const mountRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);
    sceneRef.current = scene;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const camera = new THREE.PerspectiveCamera(60, mount.clientWidth / mount.clientHeight, 0.1, 1000);
    camera.position.set(0, 1, 3);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    const light1 = new THREE.DirectionalLight(0xffffff, 1);
    light1.position.set(3, 3, 3);
    scene.add(light1);
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!renderer || !camera) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  const loadMesh = (file: File) => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    // clear previous
    for (let i = scene.children.length - 1; i >= 0; i--) {
      const child = scene.children[i];
      if (child instanceof THREE.Mesh) scene.remove(child);
    }

    const name = file.name.toLowerCase();
    const reader = new FileReader();

    reader.onload = () => {
      const contents = reader.result;
      let loader: any;

      if (name.endsWith('.gltf') || name.endsWith('.glb')) {
        loader = new GLTFLoader();
        loader.parse(contents as ArrayBuffer, '', (gltf) => {
          scene.add(gltf.scene);
        });
      } else if (name.endsWith('.obj')) {
        loader = new OBJLoader();
        const obj = loader.parse(contents as string);
        scene.add(obj);
      } else if (name.endsWith('.stl')) {
        loader = new STLLoader();
        const geometry = loader.parse(contents as ArrayBuffer);
        const material = new THREE.MeshStandardMaterial({ color: 0x8888ff });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
      } else if (name.endsWith('.ply')) {
        loader = new PLYLoader();
        const geometry = loader.parse(contents as ArrayBuffer);
        const material = new THREE.MeshStandardMaterial({ color: 0x88ff88 });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
      } else {
        alert('Unsupported file format');
        return;
      }
    };

    if (name.endsWith('.obj')) {
      reader.readAsText(file);
    } else {
      reader.readAsArrayBuffer(file);
    }
  };

  const handleFileChange = () => {
    const file = fileInputRef.current?.files?.[0];
    if (file) loadMesh(file);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div className="controls" style={{ padding: 8 }}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".obj,.ply,.stl,.gltf,.glb"
          onChange={handleFileChange}
        />
      </div>
      <div className="visualizer-container" style={{ flex: 1, width: '100%vw' }}>
        <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      </div>
    </div>
  );
}
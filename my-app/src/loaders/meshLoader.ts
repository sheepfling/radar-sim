import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {OBJLoader} from 'three/examples/jsm/loaders/OBJLoader.js';
import {PLYLoader} from 'three/examples/jsm/loaders/PLYLoader.js';
import {STLLoader} from 'three/examples/jsm/loaders/STLLoader.js';

export async function loadMeshFromFile(file: File): Promise<THREE.Object3D> {
  const ext = file.name.split('.').pop()?.toLowerCase();
  const url = URL.createObjectURL(file);
  let mesh: THREE.Object3D;
  try {
    mesh = await loadMesh(url, ext);
  } finally {
    URL.revokeObjectURL(url);
  }

  return mesh;
}


export async function loadMesh(url: string, ext?: string): Promise<THREE.Object3D> {
  ext = ext ?? url.split('.').pop()?.toLowerCase();
  return new Promise((resolve, reject) => {
    if (!ext) {
      return reject(new Error('Cannot determine file extension'));
    }
    if (ext === 'gltf' || ext === 'glb') {
      const loader = new GLTFLoader();
      loader.load(url, (gltf) => resolve(gltf.scene), undefined, reject);
    } else if (ext === 'obj') {
      const loader = new OBJLoader();
      loader.load(url, resolve, undefined, reject);
    } else if (ext === 'ply') {
      const loader = new PLYLoader();
      loader.load(url, (geometry) => {
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
          vertexColors: geometry.hasAttribute('color'),
        });
        resolve(new THREE.Mesh(geometry, material));
      }, undefined, reject);
    } else if (ext === 'stl') {
      const loader = new STLLoader();
      loader.load(url, (geometry) => {
        geometry.computeVertexNormals();
        resolve(new THREE.Mesh(geometry, undefined));
      }, undefined, reject);
    } else {
      reject(new Error(`Unsupported format: ${ext}`));
    }
  });
}

export function describeObject(obj: THREE.Object3D): string {
  const box = new THREE.Box3().setFromObject(obj);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  let vertices = 0;
  let meshes = 0;
  const materialInfo: string[] = [];

  obj.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) {
      const mesh = o as THREE.Mesh;
      const geo = mesh.geometry;
      vertices += geo.attributes.position.count;
      meshes += 1;

      const mat = mesh.material;
      if (Array.isArray(mat)) {
        mat.forEach((m, i) => materialInfo.push(`Material[${i}]: ${describeMaterial(m)}`));
      } else {
        materialInfo.push(describeMaterial(mat));
      }
    }
  });

  return [
    `Meshes: ${meshes}`,
    `Vertices: ${vertices}`,
    `Bounding Box: size=(${size.x.toFixed(3)}, ${size.y.toFixed(3)}, ${size.z.toFixed(3)})`,
    `Center: (${center.x.toFixed(3)}, ${center.y.toFixed(3)}, ${center.z.toFixed(3)})`,
    'Materials:',
    ...materialInfo
  ].join('\n');
}

export function describeMaterial(mat: THREE.Material): string {
  if ((mat as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
    const m = mat as THREE.MeshStandardMaterial;
    return [
      `type: MeshStandardMaterial`,
      `color: #${m.color.getHexString()}`,
      `metalness: ${m.metalness.toFixed(2)}`,
      `roughness: ${m.roughness.toFixed(2)}`,
      `map: ${m.map?.image?.src ?? 'none'}`,
      `metalnessMap: ${m.metalnessMap?.image?.src ?? 'none'}`,
      `roughnessMap: ${m.roughnessMap?.image?.src ?? 'none'}`,
      `normalMap: ${m.normalMap?.image?.src ?? 'none'}`,
      `emissiveMap: ${m.emissiveMap?.image?.src ?? 'none'}`
    ].join(',\n\t');
  } else if ((mat as THREE.MeshPhongMaterial).isMeshPhongMaterial) {
    const m = mat as THREE.MeshPhongMaterial;
    return [
      `type: MeshPhongMaterial`,
      `color: #${m.color.getHexString()}`,
      `specular: #${m.specular.getHexString()}`,
      `shininess: ${m.shininess}`,
      `map: ${m.map?.image?.src ?? 'none'}`,
      `normalMap: ${m.normalMap?.image?.src ?? 'none'}`
    ].join(',\n\t');
  } else {
    return `type: ${mat.type}`;
  }
}

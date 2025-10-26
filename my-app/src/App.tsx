import {useRef, useState, useEffect} from 'react'
import {Canvas, useFrame, useThree} from '@react-three/fiber'
import {OrbitControls} from '@react-three/drei'
import {STLLoader} from 'three/examples/jsm/loaders/STLLoader'
import {OBJLoader} from 'three/examples/jsm/loaders/OBJLoader'
import * as THREE from 'three'

/**
 * 3D HRRP Viewer
 * Single-file React component that:
 * - lets the user upload an OBJ/STL/GLTF
 * - renders the model with orbit controls
 * - computes a simple HRRP (range-profile) from the current camera view
 * - updates the HRRP in realtime as the camera moves
 *
 * This is an approximate, educational HRRP: it bins triangle centroids by
 * range along the view direction and weights by triangle area and
 * the face alignment to the view (|n dot v|). For better realism, add
 * physics-based RCS, high-resolution sampling, or GPU-based binning.
 */

export default function HRRPViewer() {
  const [model, setModel] = useState<THREE.Object3D | null>(null)
  const [bins, setBins] = useState(256)
  const [hrrp, setHrrp] = useState<number[]>(() => new Array(256).fill(0))
  const [rangeLimits, setRangeLimits] = useState<[number, number]>([0, 10])

  // handle file input
  async function handleFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    let obj: THREE.Object3D | null = null

    try {
      if (file.name.toLowerCase().endsWith('.stl')) {
        const loader = new STLLoader()
        const geom = loader.parse(await file.arrayBuffer())
        const mat = new THREE.MeshStandardMaterial({color: 0x888888})
        obj = new THREE.Mesh(geom, mat)
      } else if (file.name.toLowerCase().endsWith('.obj')) {
        const loader = new OBJLoader()
        const text = await file.text()
        obj = loader.parse(text)
      } else if (file.name.toLowerCase().endsWith('.gltf') || file.name.toLowerCase().endsWith('.glb')) {
        // simple gltf handling via drei useGLTF is better, but here we load manually
        // user can instead drag a gltf into the <Canvas> using a dedicated loader component
        console.warn('GLTF: use drag/drop loader in future')
      } else {
        alert('Unsupported file type: please use .stl or .obj')
      }
    } catch (e) {
      console.error(e)
      alert('Failed to load model')
    }

    if (obj) {
      // compute a tight bounding sphere to set sensible range limits
      const box = new THREE.Box3().setFromObject(obj)
      const center = box.getCenter(new THREE.Vector3())
      obj.position.sub(center) // recenter
      setModel(obj)

      const size = box.getSize(new THREE.Vector3()).length()
      setRangeLimits([0.1, Math.max(5, size * 2)])
    }
  }

  return (
    <div className="h-screen w-screen flex flex-col">
      <div className="p-2 flex gap-2 items-center">
        <input type="file" accept=".stl,.obj,.gltf,.glb" onChange={handleFile}/>
        <label>Bins</label>
        <input
          type="range"
          min={32}
          max={1024}
          value={bins}
          onChange={e => {
            const v = Number(e.target.value);
            setBins(v);
            setHrrp(new Array(v).fill(0))
          }}
        />
        <span>{bins}</span>
      </div>

      <div className="flex-1 grid grid-cols-3 gap-2 p-2">
        <div className="col-span-2 bg-black rounded">
          <Canvas camera={{position: [0, 0, 5], fov: 45}}>
            <ambientLight/>
            <pointLight position={[10, 10, 10]}/>
            <Scene
              model={model}
              bins={bins}
              onHRRPUpdate={setHrrp}
              rangeLimits={rangeLimits}
            />
            <OrbitControls/>
          </Canvas>
        </div>

        <div className="col-span-1 bg-white rounded p-2">
          <h3 className="font-bold">HRRP (range profile)</h3>
          <HRRPPlot values={hrrp}/>
          <p className="text-sm mt-2">Simple, approximate HRRP. Move camera to change LOS.</p>
        </div>
      </div>
    </div>
  )
}

/**
 * Scene: inserts the model and runs HRRP computation each frame when camera moves.
 */
function Scene({
                 model,
                 bins,
                 onHRRPUpdate,
                 rangeLimits
               }: {
  model: THREE.Object3D | null;
  bins: number;
  onHRRPUpdate: (v: number[]) => void;
  rangeLimits: [number, number]
}) {
  const groupRef = useRef<THREE.Group>(null)
  const {camera, gl} = useThree()

  // helper to compute HRRP from a mesh (triangles)
  /**
   * Compute HRRP for a THREE.Object3D by iterating its triangle meshes.
   *
   * It uses triangle centroids projected along the view (camera) direction.
   * The returned array length is `bins`.
   */
  function computeHRRP(obj: THREE.Object3D, bins: number, nearFar: [number, number]) {
    const counts = new Array(bins).fill(0)
    const minRange = nearFar[0]
    const maxRange = nearFar[1]
    const binSize = (maxRange - minRange) / bins

    const tmpVecA = new THREE.Vector3()
    const tmpVecB = new THREE.Vector3()
    const tmpVecC = new THREE.Vector3()
    const worldPos = new THREE.Vector3()
    const centroid = new THREE.Vector3()
    const normal = new THREE.Vector3()

    const viewDir = new THREE.Vector3()
    camera.getWorldDirection(viewDir)
    // viewDir is the direction camera is looking (points forward). Radar LOS is -viewDir if radar at camera position
    const radarDir = viewDir.clone().negate()
    const radarPos = camera.getWorldPosition(new THREE.Vector3())

    obj.traverse((child) => {
      const mesh = child as THREE.Mesh
      if (!mesh.isMesh) return
      const geom = mesh.geometry as THREE.BufferGeometry
      if (!geom.attributes.position) return

      const pos = geom.attributes.position.array as Float32Array
      const idx = geom.index ? (geom.index.array as Uint32Array | Uint16Array | Uint8Array) : null
      const normalAttr = geom.attributes.normal

      // iterate triangles
      const triCount = idx ? idx.length / 3 : pos.length / 9
      for (let t = 0; t < triCount; t++) {
        let aIdx: number, bIdx: number, cIdx: number
        if (idx) {
          aIdx = idx[3 * t] * 3
          bIdx = idx[3 * t + 1] * 3
          cIdx = idx[3 * t + 2] * 3
        } else {
          aIdx = (3 * t) * 3
          bIdx = (3 * t + 1) * 3
          cIdx = (3 * t + 2) * 3
        }

        tmpVecA.set(pos[aIdx], pos[aIdx + 1], pos[aIdx + 2])
        tmpVecB.set(pos[bIdx], pos[bIdx + 1], pos[bIdx + 2])
        tmpVecC.set(pos[cIdx], pos[cIdx + 1], pos[cIdx + 2])

        // transform to world
        mesh.localToWorld(tmpVecA)
        mesh.localToWorld(tmpVecB)
        mesh.localToWorld(tmpVecC)

        // centroid
        centroid.copy(tmpVecA).add(tmpVecB).add(tmpVecC).multiplyScalar(1 / 3)

        // triangle area (half cross product)
        const edge1 = tmpVecB.clone().sub(tmpVecA)
        const edge2 = tmpVecC.clone().sub(tmpVecA)
        const cross = new THREE.Vector3().copy(edge1).cross(edge2)
        const area = 0.5 * cross.length()

        // normal approx
        normal.copy(cross).normalize()

        // range along radar LOS (positive forward from radar)
        const rel = centroid.clone().sub(radarPos)
        const range = rel.dot(radarDir)

        if (range < minRange || range > maxRange) continue

        // incidence: how well the face points toward radar (abs to allow backscattering simplification)
        const inc = Math.abs(normal.dot(radarDir))

        const weight = area * inc

        // bin index
        const binIdx = Math.floor((range - minRange) / binSize)
        if (binIdx >= 0 && binIdx < bins) counts[binIdx] += weight
      }
    })

    // convert to dB-like scale for display
    const maxVal = Math.max(...counts) || 1
    const out = counts.map(v => 20 * Math.log10(v / maxVal + 1e-9))
    return out
  }

  // recompute HRRP when the camera moves (on every frame, throttle if needed)
  useFrame(() => {
    if (!model) return
    const vals = computeHRRP(model, bins, rangeLimits)
    onHRRPUpdate(vals)
  })

  return (
    <group ref={groupRef}>
      {model ? <primitive object={model}/> : <DefaultPlaceholder/>}
    </group>
  )
}

function DefaultPlaceholder() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]}/>
      <meshStandardMaterial color={0x666666}/>
    </mesh>
  )
}

/**
 * HRRPPlot: draws the HRRP values into a small canvas.
 * Keep a simple drawing; this is fast and dependency free.
 */
function HRRPPlot({values}: { values: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')
    const w = c.width = 600
    const h = c.height = 200
    ctx.clearRect(0, 0, w, h)

    // values are in dB-like negative numbers; map to 0..1
    const min = Math.min(...values)
    const max = Math.max(...values)
    const span = Math.max(1e-6, max - min)

    ctx.beginPath()
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * w
      const y = h - ((values[i] - min) / span) * h
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.strokeStyle = '#0070f3'
    ctx.lineWidth = 2
    ctx.stroke()
  }, [values])

  return <canvas ref={canvasRef} className="w-full border"/>
}

// eof

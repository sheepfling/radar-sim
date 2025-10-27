import {useRef, useState} from 'react';
import {describeObject, loadMeshFromFile, DescribedObject} from '../../loaders/meshLoader';
import React from "react";

export default function MeshLoader() {
    const inputRef = useRef<HTMLInputElement>(null);
    const [info, setInfo] = useState<DescribedObject | null>(null);

    const handleLoad = async () => {
        const file = inputRef.current?.files?.[0];
        if (!file) {
            return alert('Please select a file first.');
        }
        try {
            const obj = await loadMeshFromFile(file);
            const described = describeObject(obj);
            setInfo(described);
        } catch (err) {
            console.error(err);
            alert('Failed to load mesh.');
        }
    };

    return (
        <div style={{padding: 16}}>
            <h2>Mesh Loader Debug</h2>
            <input type="file" ref={inputRef} accept=".obj,.ply,.stl,.gltf,.glb" />
            <button onClick={handleLoad}>Load Model</button>

            {info && (
                <div style={{marginTop: 20}}>
                    <h3>Summary</h3>
                    <ul>
                        <li><strong>Meshes:</strong> {info.meshes}</li>
                        <li><strong>Vertices:</strong> {info.vertices}</li>
                    </ul>

                    <h3>Bounding Box</h3>
                    <ul>
                        <li><strong>Size:</strong> ({info.boundingBox.size.x.toFixed(3)}, {info.boundingBox.size.y.toFixed(3)}, {info.boundingBox.size.z.toFixed(3)})</li>
                        <li><strong>Center:</strong> ({info.boundingBox.center.x.toFixed(3)}, {info.boundingBox.center.y.toFixed(3)}, {info.boundingBox.center.z.toFixed(3)})</li>
                    </ul>

                    <h3>Materials</h3>
                    {info.materials.map((m, idx) => (
                        <div key={idx} style={{border: '1px solid #ccc', padding: 10, marginBottom: 10}}>
                            <p><strong>Type:</strong> {m.type}</p>
                            {m.color && <p><strong>Color:</strong> {m.color}</p>}
                            {m.metalness !== undefined && <p><strong>Metalness:</strong> {m.metalness}</p>}
                            {m.roughness !== undefined && <p><strong>Roughness:</strong> {m.roughness}</p>}
                            {m.specular && <p><strong>Specular:</strong> {m.specular}</p>}
                            {m.shininess !== undefined && <p><strong>Shininess:</strong> {m.shininess}</p>}
                            <details>
                                <summary>Texture Maps</summary>
                                <ul>
                                    {Object.entries(m.maps).map(([key, value]) => (
                                        <li key={key}>
                                            <strong>{key}:</strong> {value}
                                        </li>
                                    ))}
                                </ul>
                            </details>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
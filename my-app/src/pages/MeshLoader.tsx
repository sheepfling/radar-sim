import {useRef, useState} from 'react';
import {describeObject, loadMeshFromFile} from '../loaders/meshLoader';
import React from "react";

export default function MeshLoader() {
    const inputRef = useRef<HTMLInputElement>(null);
    const [info, setInfo] = useState('');

    const handleLoad = async () => {
        const file = inputRef.current?.files?.[0];
        if (!file) {
            return alert('Please select a file first.');
        }

        try {
            const obj = await loadMeshFromFile(file);
            setInfo(describeObject(obj));
        } catch (err) {
            console.error(err);
            alert('Failed to load mesh.');
        }
    };

    return (
        <div>
            <h2>Mesh Loader Debug</h2>
            <input type="file" ref={inputRef} accept=".obj,.ply,.stl,.gltf,.glb"/>
            <button onClick={handleLoad}>Load Model</button>
            <textarea readOnly value={info} style={{width: '100%', height: 200}}/>
        </div>
    );
}
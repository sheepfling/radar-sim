import {BrowserRouter, Routes, Route, Link} from 'react-router-dom';
import MeshLoader from './pages/debug/MeshLoader';
import MeshVisualizer from './pages/debug/MeshVisualizer';
import Home from './pages/Home';
import React from "react";

export default function App() {
    return (
        <BrowserRouter>
            <nav>
                <Link to="/">Home</Link> | <Link to="/debug/mesh_loader">Debug Page</Link>
            </nav>
            <Routes>
                <Route path="/" element={<Home/>}/>
                <Route path="/debug/mesh_loader" element={<MeshLoader/>}/>
                <Route path="/debug/mesh_visualizer" element={<MeshVisualizer/>}/>
            </Routes>
        </BrowserRouter>
    );
}
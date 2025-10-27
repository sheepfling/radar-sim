import {BrowserRouter, Routes, Route, Link} from 'react-router-dom';
import MeshLoader from './pages/MeshLoader';
import Home from './pages/Home';
import React from "react";

export default function App() {
    return (
        <BrowserRouter>
            <nav>
                <Link to="/">Home</Link> | <Link to="/debug">Debug Page</Link>
            </nav>
            <Routes>
                <Route path="/" element={<Home/>}/>
                <Route path="/MeshLoader" element={<MeshLoader/>}/>
            </Routes>
        </BrowserRouter>
    );
}
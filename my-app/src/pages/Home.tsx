import React from 'react';
import {Link} from 'react-router-dom';

export default function Home() {
    return (
        <div style={{padding: 20}}>
            <h1>Welcome to the Radar Simulator</h1>
            <p>
                This is the main landing page. From here, you can navigate to different sections:
            </p>
            <ul>
                <li>
                    <Link to="/debug">Debug / Mesh Loader</Link>
                </li>
                {/* Add more links here if you have other pages */}
            </ul>
        </div>
    );
}
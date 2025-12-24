// Three.js 3D Neon Logo
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('logo-container');
    if (!container) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 200, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, 200);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // Create "Z" shape geometry
    const shape = new THREE.Shape();
    shape.moveTo(-3, -3);
    shape.lineTo(3, -3);
    shape.lineTo(-3, 3);
    shape.lineTo(3, 3);

    const geometry = new THREE.ExtrudeGeometry(shape, {
        depth: 0.5,
        bevelEnabled: true,
        bevelThickness: 0.2,
        bevelSize: 0.1,
        bevelSegments: 3
    });

    // Neon cyan material
    const material = new THREE.MeshPhongMaterial({
        color: 0x00ffff,
        emissive: 0x00aaaa,
        emissiveIntensity: 2,
        shininess: 100
    });

    const logo = new THREE.Mesh(geometry, material);
    scene.add(logo);

    // Add glow effect
    const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ffff,
        transparent: true,
        opacity: 0.3
    });
    
    const glowMesh = new THREE.Mesh(geometry.clone().scale(1.1, 1.1, 1.1), glowMaterial);
    scene.add(glowMesh);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0x00ffff, 2, 100);
    pointLight1.position.set(10, 10, 10);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xff00ff, 2, 100);
    pointLight2.position.set(-10, -10, 10);
    scene.add(pointLight2);

    camera.position.z = 10;

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        
        // Rotate slowly
        logo.rotation.x = Math.sin(Date.now() * 0.0005) * 0.3;
        logo.rotation.y += 0.005;
        glowMesh.rotation.copy(logo.rotation);
        
        // Pulsate emissive intensity
        material.emissiveIntensity = 1.5 + Math.sin(Date.now() * 0.003) * 0.5;
        
        renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / 200;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, 200);
    });
});


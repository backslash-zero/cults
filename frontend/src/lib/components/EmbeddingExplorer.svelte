<script lang="ts">
	import * as THREE from 'three';
	import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
	import type { PointMeta, SourceDataset } from '$lib/types';
	import { colorForDataset } from '$lib/pointColors';

	let {
		points,
		positions,
		visible,
		searchTerm,
		onSelect
	}: {
		points: PointMeta[];
		positions: [number, number, number][];
		visible: Record<SourceDataset, boolean>;
		searchTerm: string;
		onSelect: (point: PointMeta | null) => void;
	} = $props();

	// The canvas always renders on a fixed dark background regardless of the
	// page's light/dark theme -- three of the six categorical colors are
	// documented as sub-3:1 contrast against a light surface, and a dark
	// "viewport" is the standard convention for point-cloud/scatter
	// visualizations like this one (e.g. TensorFlow Projector). Point colors
	// always use each category's dark-surface color step accordingly.
	const CANVAS_BACKGROUND = '#0d0d0d';
	const HIGHLIGHT_COLOR = '#ffffff';

	let containerEl: HTMLDivElement;
	let canvasEl: HTMLCanvasElement;

	let scene: THREE.Scene;
	let camera: THREE.PerspectiveCamera;
	let renderer: THREE.WebGLRenderer;
	let controls: OrbitControls;
	let pointCloud: THREE.Points | null = null;
	let visibleIndices: number[] = [];
	let raycastThreshold = 0.5;
	let animationFrameId: number;

	const raycaster = new THREE.Raycaster();
	const DIM_MIX = 0.85; // how far non-matches move toward the background when searching

	// One-time scene/camera/renderer/controls setup, tied to the canvas
	// becoming available. Re-runs only if canvasEl itself changes (it won't,
	// in practice, for the lifetime of this component instance).
	$effect(() => {
		if (!canvasEl || !containerEl) return;

		scene = new THREE.Scene();
		scene.background = new THREE.Color(CANVAS_BACKGROUND);
		camera = new THREE.PerspectiveCamera(60, 1, 0.1, 5000);
		camera.position.set(0, 0, 50);

		renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

		controls = new OrbitControls(camera, renderer.domElement);
		controls.enableDamping = true;
		controls.dampingFactor = 0.08;

		function resize() {
			const { clientWidth, clientHeight } = containerEl;
			if (clientWidth === 0 || clientHeight === 0) return;
			camera.aspect = clientWidth / clientHeight;
			camera.updateProjectionMatrix();
			renderer.setSize(clientWidth, clientHeight);
		}
		resize();
		const resizeObserver = new ResizeObserver(resize);
		resizeObserver.observe(containerEl);

		function animate() {
			animationFrameId = requestAnimationFrame(animate);
			controls.update();
			renderer.render(scene, camera);
		}
		animate();

		function onClick(event: MouseEvent) {
			if (!pointCloud) return;
			const rect = canvasEl.getBoundingClientRect();
			const ndc = new THREE.Vector2(
				((event.clientX - rect.left) / rect.width) * 2 - 1,
				-((event.clientY - rect.top) / rect.height) * 2 + 1
			);
			raycaster.setFromCamera(ndc, camera);
			raycaster.params.Points!.threshold = raycastThreshold;
			const hits = raycaster.intersectObject(pointCloud);
			if (hits.length > 0 && hits[0].index !== undefined) {
				const originalIndex = visibleIndices[hits[0].index];
				onSelect(points[originalIndex] ?? null);
			}
		}
		canvasEl.addEventListener('click', onClick);

		return () => {
			cancelAnimationFrame(animationFrameId);
			resizeObserver.disconnect();
			canvasEl.removeEventListener('click', onClick);
			controls.dispose();
			renderer.dispose();
			pointCloud?.geometry.dispose();
			(pointCloud?.material as THREE.Material | undefined)?.dispose();
		};
	});

	// Refit the camera whenever the projection method changes (a new
	// `positions` array reference), using the full unfiltered set so the
	// framing doesn't jump around as categories are toggled.
	$effect(() => {
		if (!scene || !camera || !controls || positions.length === 0) return;

		const box = new THREE.Box3();
		for (const [x, y, z] of positions) {
			box.expandByPoint(new THREE.Vector3(x, y, z));
		}
		const center = box.getCenter(new THREE.Vector3());
		const size = box.getSize(new THREE.Vector3());
		const radius = Math.max(size.x, size.y, size.z, 1) / 2;

		camera.position.set(center.x, center.y, center.z + radius * 2.5);
		camera.near = radius / 100;
		camera.far = radius * 100;
		camera.updateProjectionMatrix();
		controls.target.copy(center);
		controls.update();

		raycastThreshold = radius / 80;
	});

	// Rebuild the point-cloud geometry whenever the visible dataset, points,
	// positions, search term, or theme changes.
	$effect(() => {
		if (!scene) return;

		const query = searchTerm.trim().toLowerCase();
		const highlightColor = new THREE.Color(HIGHLIGHT_COLOR);
		const dimTarget = new THREE.Color(CANVAS_BACKGROUND);
		const filteredPositions: number[] = [];
		const filteredColors: number[] = [];
		visibleIndices = [];

		for (let i = 0; i < points.length; i++) {
			const point = points[i];
			if (!visible[point.source_dataset]) continue;

			const coord = positions[i];
			if (!coord) continue;

			visibleIndices.push(i);
			filteredPositions.push(coord[0], coord[1], coord[2]);

			const baseColor = new THREE.Color(colorForDataset(point.source_dataset, true));
			if (query.length > 0) {
				const isMatch = point.label.toLowerCase().includes(query);
				const color = isMatch ? highlightColor : baseColor.clone().lerp(dimTarget, DIM_MIX);
				filteredColors.push(color.r, color.g, color.b);
			} else {
				filteredColors.push(baseColor.r, baseColor.g, baseColor.b);
			}
		}

		if (pointCloud) {
			scene.remove(pointCloud);
			pointCloud.geometry.dispose();
			(pointCloud.material as THREE.Material).dispose();
			pointCloud = null;
		}

		if (visibleIndices.length === 0) return;

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute(filteredPositions, 3));
		geometry.setAttribute('color', new THREE.Float32BufferAttribute(filteredColors, 3));

		const material = new THREE.PointsMaterial({ size: 0.35, vertexColors: true, sizeAttenuation: true });
		pointCloud = new THREE.Points(geometry, material);
		scene.add(pointCloud);
	});
</script>

<div bind:this={containerEl} class="w-full h-full">
	<canvas bind:this={canvasEl} class="w-full h-full block cursor-pointer"></canvas>
</div>

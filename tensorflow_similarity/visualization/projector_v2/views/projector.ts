/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * Renders embedding visualization in a 3d scene.
 */

import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';

import {Messenger} from '../lib/ipc';
import {createElement, updateStyles} from '../lib/renderer';
import {InboundMessageType, Message, Metadata, Point} from '../lib/types';
import {SettingsPanel} from './settings';

const HOVERED_COLOR = new THREE.Color(0xff0000);
const DEFAULT_OPACITY = 0.6;
const anyWindow = window;

export class Projector {
  private readonly settings = new SettingsPanel();
  private readonly renderer = new THREE.WebGLRenderer();
  private readonly scene = new THREE.Scene();
  private readonly pointGroup = new THREE.Group();
  private camera?: THREE.Camera;
  private control?: OrbitControls;
  private readonly raycaster = new THREE.Raycaster();

  private metadata: Metadata[] = [];
  private readonly objectToIndex = new WeakMap<THREE.Object3D, number>();
  private readonly container: HTMLElement;
  private readonly labelImg = createElement('div');
  private readonly labelText = createElement('div');
  private readonly labelContainer = createElement('div', null, [
    this.labelText,
    this.labelImg,
  ]);
  private containerRect?: DOMRect;

  private hoveredMesh: THREE.Mesh | null = null;

  constructor(messenger: Messenger) {
    messenger.addOnMessage((message: Message) => {
      switch (message.type) {
        case InboundMessageType.UPDATE: {
          const updateMessage = message;
          this.updatePoints(updateMessage.mainPayload);
          break;
        }
        case InboundMessageType.METADATA: {
          const metadataMessage = message;
          this.setMetadata(metadataMessage.mainPayload);
          break;
        }
      }
    });

    const {canvas, container} = this.setupDom();
    this.container = container;

    this.renderer.setClearColor(0xffffff, 1);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    this.scene.add(this.pointGroup);
    this.scene.add(new THREE.AxesHelper(Number.MAX_SAFE_INTEGER));

    this.settings.addSettingChangeListener((prev, next) => {
      if (!prev || prev.threeDimensions !== next.threeDimensions) {
        if (this.camera) {
          this.scene.remove(this.camera);
        }

        const {width, height} = container.getBoundingClientRect();
        if (next.threeDimensions) {
          this.camera = new THREE.PerspectiveCamera(
            55,
            width / height,
            1,
            20000
          );
          this.camera.position.set(30, 30, 100);
        } else {
          this.camera = new THREE.OrthographicCamera(
            -width / 2,
            width / 2,
            height / 2,
            -height / 2,
            -Number.MAX_SAFE_INTEGER,
            Number.MAX_SAFE_INTEGER
          );
        }
        this.scene.add(this.camera);

        if (!this.control) {
          this.control = new OrbitControls(this.camera, canvas);
        } else {
          this.control.object = this.camera;
        }
      }
    });

    anyWindow.requestIdleCallback(() => this.animate());
  }

  private setupDom(): {
    canvas: HTMLCanvasElement;
    container: HTMLElement;
  } {
    const canvas = this.renderer.domElement;
    updateStyles(this.labelContainer, {
      position: 'absolute',
      pointerEvents: 'none',
      willChange: 'top, left',
    } as Partial<CSSStyleDeclaration>);
    updateStyles(this.labelImg, {
      width: '300px',
      height: '200px',
      backgroundSize: 'contain',
      backgroundRepeat: 'no-repeat',
    });

    const container = createElement('div', null, [
      this.renderer.domElement,
      this.labelContainer,
      this.settings.getDomElement(),
    ]);
    updateStyles(container, {
      height: '100%',
      overflow: 'hidden',
      position: 'relative',
      width: '100%',
    });

    const resizeObserver = new anyWindow.ResizeObserver(() => {
      this.containerRect = container.getBoundingClientRect();
      const {width, height} = this.containerRect;
      this.renderer.setSize(width, height);

      if (this.camera instanceof THREE.PerspectiveCamera) {
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
      }
    });
    resizeObserver.observe(container);

    canvas.addEventListener('mousemove', (event) => this.onMouseMove(event), {
      passive: true,
    });

    return {canvas, container};
  }

  getDomElement(): HTMLElement {
    return this.container;
  }

  private updatePoints(points: Point[]): void {
    const existingChildren = [...this.pointGroup.children] as THREE.Mesh[];

    existingChildren
      .slice(points.length)
      .forEach((obj) => this.pointGroup.remove(obj));

    for (const [index, point] of points.entries()) {
      let mesh = existingChildren[index];
      const x = point[0];
      const y = point[1];
      const z = point[2] ?? 0;
      const meta = this.metadata[index];

      if (mesh) {
        mesh.position.set(x, y, z);
      } else {
        const geom = new THREE.SphereBufferGeometry(1, 32, 32);
        const material = new THREE.MeshBasicMaterial({transparent: true});
        mesh = new THREE.Mesh(geom, material);
        mesh.position.set(x, y, z);
        this.updateColor(
          mesh,
          new THREE.Color(meta?.color ?? '#f00'),
          DEFAULT_OPACITY
        );
        this.pointGroup.add(mesh);
      }

      this.objectToIndex.set(mesh, index);
    }
  }

  private setMetadata(metadata: Metadata[]) {
    this.metadata = metadata;
  }

  private animate() {
    requestAnimationFrame(() => this.animate());
    if (!this.camera || !this.control) return;
    this.control.update();
    this.updateLabel();
    this.renderer.render(this.scene, this.camera);
  }

  private updateColor(
    mesh: THREE.Mesh,
    color: THREE.Color,
    opacity: number
  ): void {
    const material = mesh.material;
    if (material instanceof THREE.MeshBasicMaterial) {
      material.color = color;
      material.opacity = opacity;
      material.needsUpdate = true;
    }
  }

  private updateLabel() {
    const pointIndex = this.hoveredMesh
      ? this.objectToIndex.get(this.hoveredMesh) ?? -1
      : null;
    if (
      !this.containerRect ||
      !this.hoveredMesh ||
      pointIndex == null ||
      !this.camera
    ) {
      updateStyles(this.labelContainer, {display: 'none'});
      return;
    }

    const metadata = this.metadata[pointIndex] || {label: 'Unknown'};
    const target = this.hoveredMesh.position.clone();
    target.project(this.camera);
    const {width, height} = this.containerRect;
    const x = (target.x * width) / 2 + width / 2;
    const y = -((target.y * height) / 2) + height / 2;
    this.labelText.textContent = metadata.label;

    if (metadata.imageLabel) {
      updateStyles(this.labelImg, {
        backgroundImage: `url(${metadata.imageLabel})`,
      });
    }

    updateStyles(this.labelContainer, {
      display: 'block',
      fontSize: '1.5em',
      left: String(x + 5) + 'px',
      top: String(y + 5) + 'px',
    });
  }

  private onMouseMove(event: MouseEvent) {
    // Raycaster expects the mouse to be a number between -1 and 1.
    if (!this.containerRect || !this.camera) return;
    const clientXInDom = event.clientX - this.containerRect.x;
    const clientYInDom = event.clientY - this.containerRect.y;
    const mouse = new THREE.Vector2(
      (clientXInDom / this.containerRect.width) * 2 - 1,
      -(clientYInDom / this.containerRect.height) * 2 + 1
    );

    this.raycaster.setFromCamera(mouse, this.camera);

    // Intersections pick up things like AxesHelper.
    const intersects = this.raycaster.intersectObjects(
      this.pointGroup.children
    );
    const firstMesh = intersects.find(
      (intersection) => intersection.object instanceof THREE.Mesh
    );

    if (this.hoveredMesh === firstMesh?.object) return;
    if (this.hoveredMesh) {
      const pointIndex = this.objectToIndex.get(this.hoveredMesh) ?? -1;
      const metadata = this.metadata[pointIndex];
      if (metadata) {
        this.updateColor(
          this.hoveredMesh,
          new THREE.Color(metadata.color ?? '#f00'),
          DEFAULT_OPACITY
        );
      }
    }

    if (firstMesh) {
      this.hoveredMesh = firstMesh.object as THREE.Mesh;
      this.updateColor(this.hoveredMesh, HOVERED_COLOR, 1);
    } else {
      this.hoveredMesh = null;
    }
  }
}

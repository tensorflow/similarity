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
 * Entry point for rendering the embedding projector.
 */
import {GlobalMessenger, getCellMessenger} from './lib/ipc';
import {createElement, updateStyles} from './lib/renderer';
import {Header} from './views/header';
import {Projector} from './views/projector';

declare namespace globalThis {
  let messenger: GlobalMessenger|undefined;
  let bootstrap: ((domId: string) => void|undefined);
}

function bootstrap(domId: string) {
  const messenger = getCellMessenger();
  const main = updateStyles(
    createElement('div', null, [
      new Header(messenger).getDomElement(),
      new Projector(messenger).getDomElement(),
    ]),
    {
      display: 'grid',
      height: '100%',
      width: '100%',
    }
  );

  const bootstrapableDom = document.getElementById(domId);

  if (!bootstrapableDom) {
    throw new Error("Expected to be bootstrapped with a container created by the widget.");
  }
  bootstrapableDom.appendChild(main);
}

globalThis.messenger = globalThis.messenger || new GlobalMessenger();
globalThis.bootstrap = globalThis.bootstrap || bootstrap;

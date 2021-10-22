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

import {createElement, updateStyles} from '../lib/renderer';

export interface Settings {
  threeDimensions: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  threeDimensions: true,
};

export class SettingsPanel {
  private readonly settings = DEFAULT_SETTINGS;
  private readonly listeners = new Set<
    (prev: Settings | null, next: Settings) => void
  >();

  private readonly toggleThreeDimensions = createElement('input', {
    id: 'three-dim',
    type: 'checkbox',
    checked: this.settings.threeDimensions ? 'true' : 'false',
    onChange: (event) => this.onDimensionChange(event),
  });

  private readonly container = updateStyles(
    createElement('div', null, [
      updateStyles(
        createElement('ul', null, [
          updateStyles(
            createElement('li', null, [
              this.toggleThreeDimensions,
              createElement('label', {for: 'three-dim'}, ['3D']),
            ]),
            {
              display: 'flex',
              padding: '3px 5px',
            }
          ),
        ]),
        {
          listStyleType: 'none',
          margin: '0',
          padding: '0',
        }
      ),
    ]),
    {
      backgroundColor: '#333e',
      color: '#fff',
      minWidth: '100px',
      position: 'absolute',
      right: '0',
      top: '0',
    }
  );

  addSettingChangeListener(
    cb: (prev: Settings | null, next: Settings) => void
  ) {
    this.listeners.add(cb);
    setTimeout(() => cb(null, this.settings), 0);
  }

  getDomElement(): HTMLElement {
    return this.container;
  }

  private onDimensionChange(event: Event): void {
    const checkbox = event.target as HTMLInputElement;
    this.setSettingsAndNotify({threeDimensions: checkbox.checked});
  }

  private setSettingsAndNotify(settings: Partial<Settings>) {
    const prevSetting = {...this.settings};
    Object.assign(this.settings, settings);
    for (const cb of this.listeners) {
      cb(prevSetting, this.settings);
    }
  }
}

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

interface Listeners {
  onPassiveClick: (event: MouseEvent) => void;
  onPassiveMouseMove: (event: MouseEvent) => void;
  onChange: (event: Event) => void;
}

/**
 * Possible props to the createElement.
 */
export type Props = Exclude<
  {
    [prop: string]: string | Function;
  },
  keyof Listeners
> &
  Listeners;

/**
 * Basic element renderer that mimics API of React without JSX.
 */
export function createElement(
  tagName: string,
  props: Partial<Props> | null = null,
  children: Array<HTMLElement | string> = []
) {
  const element = document.createElement(tagName);

  if (props !== null) {
    for (const [key, value] of Object.entries(props)) {
      switch (key) {
        case 'onPassiveClick':
          element.addEventListener('click', props.onPassiveClick!, {
            passive: true,
          });
          break;
        case 'onPassiveMouseMove':
          element.addEventListener('mousemove', props.onPassiveMouseMove!, {
            passive: true,
          });
          break;
        case 'onChange':
          element.addEventListener('change', props.onChange!);
          break;
        default:
          if (typeof value === 'string') {
            element.setAttribute(key, value);
          } else {
            throw new RangeError(`Callback for ${key} is not implemented yet.`);
          }
      }
    }
  }

  for (const child of children) {
    const el =
      child instanceof HTMLElement ? child : document.createTextNode(child);
    element.appendChild(el);
  }

  return element;
}

/**
 * Applies style object onto the passed `element`.
 */
export function updateStyles(
  element: HTMLElement,
  styleObject: Partial<CSSStyleDeclaration>
): HTMLElement {
  Object.assign(element.style, styleObject);
  return element;
}

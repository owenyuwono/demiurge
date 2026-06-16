import GUI, { Controller } from 'lil-gui'

const STYLE = `
.demiurge-tabbed {
  position: fixed;
  top: 0;
  right: 0;
  z-index: 1000;
  font-family: var(--font-family, monospace);
}

.demiurge-tab-bar {
  display: flex;
  background: #1f1f1f;
  border-bottom: 1px solid #444;
}

.demiurge-tab-bar button {
  flex: 1;
  padding: 5px 8px;
  background: #1f1f1f;
  color: #ebebeb;
  border: none;
  border-right: 1px solid #444;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font-family, monospace);
  letter-spacing: 0.03em;
  transition: background 0.1s;
}

.demiurge-tab-bar button:last-child {
  border-right: none;
}

.demiurge-tab-bar button:hover {
  background: #2c2c2c;
}

.demiurge-tab-bar button.active {
  background: #3c3c3c;
  border-bottom: 2px solid #82aaff;
  color: #fff;
}

.demiurge-tab-pane {
  display: none;
}

.demiurge-tab-pane.active {
  display: block;
}

.demiurge-tabbed .lil-gui {
  --background-color: #1f1f1f;
  --text-color: #ebebeb;
  --title-background-color: #111;
  --title-text-color: #ebebeb;
  --widget-color: #424242;
  --hover-color: #4f4f4f;
  --focus-color: #595959;
  --number-color: #2cc9ff;
  --string-color: #a2db3c;
  border-radius: 0;
  box-shadow: none;
}
`

export class TabbedGui {
  readonly planetGui: GUI
  readonly atmosphereGui: GUI
  readonly viewGui: GUI

  private readonly _outer: HTMLDivElement
  private readonly _styleEl: HTMLStyleElement
  private readonly _panes: HTMLDivElement[]
  private readonly _buttons: HTMLButtonElement[]
  private _activeIndex: number = 0

  constructor(labels: [string, string, string] = ['Planet', 'Atmosphere', 'View']) {
    // Inject CSS
    this._styleEl = document.createElement('style')
    this._styleEl.textContent = STYLE
    document.head.appendChild(this._styleEl)

    // Outer container
    this._outer = document.createElement('div')
    this._outer.className = 'demiurge-tabbed'

    // Tab bar
    const tabBar = document.createElement('div')
    tabBar.className = 'demiurge-tab-bar'

    // 3 panes + 3 GUIs
    this._panes = []
    this._buttons = []

    const pane0 = this._makePane()
    const pane1 = this._makePane()
    const pane2 = this._makePane()
    this._panes.push(pane0, pane1, pane2)

    this.planetGui = new GUI({ autoPlace: false, title: labels[0] })
    this.atmosphereGui = new GUI({ autoPlace: false, title: labels[1] })
    this.viewGui = new GUI({ autoPlace: false, title: labels[2] })

    pane0.appendChild(this.planetGui.domElement)
    pane1.appendChild(this.atmosphereGui.domElement)
    pane2.appendChild(this.viewGui.domElement)

    // Build buttons
    for (let i = 0; i < 3; i++) {
      const btn = document.createElement('button')
      btn.textContent = labels[i]
      const idx = i
      btn.addEventListener('click', () => this._selectTab(idx))
      this._buttons.push(btn)
      tabBar.appendChild(btn)
    }

    this._outer.appendChild(tabBar)
    this._outer.appendChild(pane0)
    this._outer.appendChild(pane1)
    this._outer.appendChild(pane2)

    document.body.appendChild(this._outer)

    // Activate first tab
    this._selectTab(0)
  }

  private _makePane(): HTMLDivElement {
    const div = document.createElement('div')
    div.className = 'demiurge-tab-pane'
    return div
  }

  private _selectTab(index: number): void {
    this._activeIndex = index
    for (let i = 0; i < 3; i++) {
      const isActive = i === index
      this._panes[i].classList.toggle('active', isActive)
      this._buttons[i].classList.toggle('active', isActive)
    }
  }

  controllersRecursive(): Controller[] {
    return [
      ...this.planetGui.controllersRecursive(),
      ...this.atmosphereGui.controllersRecursive(),
      ...this.viewGui.controllersRecursive(),
    ]
  }

  destroy(): void {
    this.planetGui.destroy()
    this.atmosphereGui.destroy()
    this.viewGui.destroy()
    this._outer.remove()
    this._styleEl.remove()
  }
}

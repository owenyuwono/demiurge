import { NavMode } from '../controls/NavMode.js'

export interface NavControlsCallbacks {
  onEnterPlacement: () => void
  onReset: () => void
}

const STYLE_ID = 'nav-controls-style'

const BUTTON_BASE: Partial<CSSStyleDeclaration> = {
  padding: '6px 14px',
  fontSize: '13px',
  fontFamily: 'monospace',
  border: '1px solid rgba(255,255,255,0.35)',
  borderRadius: '4px',
  background: 'rgba(0,0,0,0.55)',
  color: '#eee',
  cursor: 'pointer',
  transition: 'opacity 0.15s',
}

function applyStyle(el: HTMLElement, styles: Partial<CSSStyleDeclaration>): void {
  for (const [key, value] of Object.entries(styles)) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(el.style as any)[key] = value
  }
}

export class NavControlsBar {
  private readonly container: HTMLDivElement
  private readonly walkBtn: HTMLButtonElement
  private readonly resetBtn: HTMLButtonElement
  private readonly styleEl: HTMLStyleElement | null

  private readonly onWalkClick: () => void
  private readonly onResetClick: () => void

  constructor(callbacks: NavControlsCallbacks) {
    // Inject a tiny stylesheet once (shared across instances, though only one is expected)
    if (!document.getElementById(STYLE_ID)) {
      const style = document.createElement('style')
      style.id = STYLE_ID
      style.textContent = `
        #nav-controls button:disabled {
          opacity: 0.4;
          cursor: default;
        }
        #nav-controls button:not(:disabled):hover {
          background: rgba(255,255,255,0.18);
        }
      `
      document.head.appendChild(style)
      this.styleEl = style
    } else {
      this.styleEl = null
    }

    // Container
    const container = document.createElement('div')
    container.id = 'nav-controls'
    applyStyle(container, {
      position: 'fixed',
      bottom: '24px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: '100',
      display: 'flex',
      gap: '10px',
      padding: '8px 12px',
      background: 'rgba(0,0,0,0.45)',
      borderRadius: '6px',
      border: '1px solid rgba(255,255,255,0.15)',
      backdropFilter: 'blur(4px)',
    })

    // "Walk here" button
    const walkBtn = document.createElement('button')
    walkBtn.type = 'button'
    walkBtn.textContent = 'Walk here'
    applyStyle(walkBtn, BUTTON_BASE)

    // "Back to orbit" button
    const resetBtn = document.createElement('button')
    resetBtn.type = 'button'
    resetBtn.textContent = 'Back to orbit'
    applyStyle(resetBtn, BUTTON_BASE)

    // Store bound handlers so we can remove them in dispose()
    this.onWalkClick = () => callbacks.onEnterPlacement()
    this.onResetClick = () => callbacks.onReset()

    walkBtn.addEventListener('click', this.onWalkClick)
    resetBtn.addEventListener('click', this.onResetClick)

    container.appendChild(walkBtn)
    container.appendChild(resetBtn)
    document.body.appendChild(container)

    this.container = container
    this.walkBtn = walkBtn
    this.resetBtn = resetBtn

    // Start in Globe mode
    this.setMode(NavMode.Globe)
  }

  setMode(mode: NavMode): void {
    switch (mode) {
      case NavMode.Globe:
        this.walkBtn.textContent = 'Walk here'
        this.walkBtn.disabled = false
        this.resetBtn.disabled = true
        break

      case NavMode.Placement:
        this.walkBtn.textContent = 'Click the planet…'
        this.walkBtn.disabled = true
        this.resetBtn.disabled = false
        break

      case NavMode.FirstPerson:
        this.walkBtn.textContent = 'Walk here'
        this.walkBtn.disabled = true
        this.resetBtn.disabled = false
        break
    }
  }

  dispose(): void {
    this.walkBtn.removeEventListener('click', this.onWalkClick)
    this.resetBtn.removeEventListener('click', this.onResetClick)

    if (this.container.parentNode) {
      this.container.parentNode.removeChild(this.container)
    }

    // Only remove the shared style element if this instance injected it
    if (this.styleEl && this.styleEl.parentNode) {
      this.styleEl.parentNode.removeChild(this.styleEl)
    }
  }
}

const KEYMAP_HELP = `
─── planet ──────────────────
1  wireframe
2  LOD colors
3  tectonics
4  spin
f  freeze LOD
g  new seed

─── camera ──────────────────
drag      rotate globe
wheel     zoom
WASD      orbit
shift     fast`.trimStart()

export class Hud {
  private readonly el: HTMLElement

  constructor(container?: HTMLElement) {
    if (container) {
      this.el = container
    } else {
      const existing = document.getElementById('hud')
      if (existing) {
        this.el = existing
      } else {
        const div = document.createElement('div')
        div.id = 'hud'
        document.body.appendChild(div)
        this.el = div
      }
    }
  }

  update(fields: Record<string, string>): void {
    const lines = Object.entries(fields)
      .map(([k, v]) => `${k.padEnd(14)}${v}`)
      .join('\n')

    this.el.textContent = lines + '\n\n' + KEYMAP_HELP
  }
}

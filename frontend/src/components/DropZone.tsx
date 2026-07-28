import { useRef, useState, type ReactNode } from 'react'

const ACCEPTED = ['.pdf', '.docx', '.html', '.md', '.txt']

function accepted(file: File): boolean {
  const name = file.name.toLowerCase()
  return ACCEPTED.some((ext) => name.endsWith(ext))
}

export function DropZone({
  onFiles,
  children,
}: {
  onFiles: (files: File[]) => void
  children: ReactNode
}) {
  const [over, setOver] = useState(false)
  const [rejected, setRejected] = useState<string | null>(null)

  // dragenter/dragleave also fire when the pointer crosses *child* elements, so
  // a naive boolean flickers the overlay on every internal boundary. Counting
  // enters against leaves is the reliable way to know when the cursor has
  // actually left the container.
  const depth = useRef(0)

  function reset() {
    depth.current = 0
    setOver(false)
  }

  return (
    <div
      // min-h-0: flex items refuse to shrink below their content by default,
      // which would let a long conversation grow this column past the viewport
      // and turn the page into the scroll container.
      className="relative flex min-h-0 flex-1 flex-col"
      onDragEnter={(e) => {
        e.preventDefault()
        if (!e.dataTransfer.types.includes('Files')) return
        depth.current += 1
        setOver(true)
        setRejected(null)
      }}
      onDragOver={(e) => {
        // Without preventDefault the browser navigates to the dropped file.
        e.preventDefault()
        e.dataTransfer.dropEffect = 'copy'
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        depth.current -= 1
        if (depth.current <= 0) reset()
      }}
      onDrop={(e) => {
        e.preventDefault()
        reset()

        const all = Array.from(e.dataTransfer.files)
        const ok = all.filter(accepted)
        const bad = all.filter((f) => !accepted(f))

        if (bad.length > 0) {
          setRejected(
            `Skipped ${bad.map((f) => f.name).join(', ')} — supported types: ${ACCEPTED.join(', ')}`,
          )
        }
        if (ok.length > 0) onFiles(ok)
      }}
    >
      {children}

      {over && (
        <div className="pointer-events-none absolute inset-3 z-20 flex items-center justify-center rounded-xl border-2 border-dashed border-indigo-500 bg-indigo-50/85 backdrop-blur-[1px] dark:bg-indigo-950/70">
          <div className="text-center">
            <p className="text-sm font-medium text-indigo-700 dark:text-indigo-200">
              Drop to upload
            </p>
            <p className="mt-1 text-xs text-indigo-600/80 dark:text-indigo-300/80">
              {ACCEPTED.join('  ·  ')}
            </p>
          </div>
        </div>
      )}

      {rejected && (
        <div className="absolute inset-x-0 bottom-28 z-20 mx-auto w-fit rounded-lg bg-amber-100 px-3 py-2 text-xs text-amber-900 shadow dark:bg-amber-950 dark:text-amber-200">
          {rejected}
          <button
            onClick={() => setRejected(null)}
            className="ml-2 font-medium underline"
          >
            dismiss
          </button>
        </div>
      )}
    </div>
  )
}

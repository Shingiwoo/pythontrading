<script>
  import { onMount } from 'svelte'
  let status = {}
  let symbol = 'BTCUSDT'

  async function loadStatus() {
    const res = await fetch('/api/status')
    status = await res.json()
  }

  async function start() {
    await fetch('/api/bot/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol })
    })
    loadStatus()
  }

  async function stop() {
    await fetch('/api/bot/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol })
    })
    loadStatus()
  }

  onMount(loadStatus)
</script>

<h1>Bot Trading</h1>
<label>Simbol: <input bind:value={symbol} /></label>
<button on:click={start}>Mulai</button>
<button on:click={stop}>Stop</button>

<pre>{JSON.stringify(status, null, 2)}</pre>

<template>
  <div>
    <div class="canvas-wrapper" ref="canvasWrapper">
      <div class="draw-area">
        <canvas id="canvas" ref="canvas" :width="width" :height="height"></canvas>
        <canvas id="cursor" ref="cursor" :width="width" :height="height"></canvas>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CanvasDraw',
  props: {
    brushSize: {
      type: Number,
      default: 12,
    },
    width: {
      type: Number,
      default: 240,
    },
    height: {
      type: Number,
      default: 240,
    },
  },
  data() {
    return {
      canvasContext: null,
      cursorContext: null,
      isDrawing: false,
      lastX: 0,
      lastY: 0,
      tools: [
        {
          name: 'Pencil',
          color: '#555',
        },
      ],
      selectedToolIdx: 0,
    };
  },
  mounted() {
    this.setCanvas();
    this.bindEvents();
  },
  methods: {
    setCanvas() {
      this.$refs.canvasWrapper.style.gridTemplateColumns = `${this.width}px 30px`;
      this.$refs.canvasWrapper.style.width = `${this.width + 30}px`;
      this.$refs.canvasWrapper.style.height = `${this.height}px`;
      this.canvasContext = this.$refs.canvas.getContext('2d');
      this.canvasContext.lineJoin = 'round';
      this.canvasContext.lineCap = 'round';
      this.canvasContext.lineWidth = this.brushSize;
      this.canvasContext.strokeStyle = this.tools[this.selectedToolIdx].color;
      this.cursorContext = this.$refs.cursor.getContext('2d');
    },
    bindEvents() {
      this.$refs.canvas.addEventListener('mousedown', (event) => {
        this.isDrawing = true;
        [this.lastX, this.lastY] = [event.offsetX, event.offsetY];
      });
      this.$refs.canvas.addEventListener('mousemove', this.draw);
      this.$refs.canvas.addEventListener('mouseup', () => this.isDrawing = false);
      this.$refs.canvas.addEventListener('mouseout', () => this.isDrawing = false);
    },
    draw(event) {
      this.drawCursor(event);
      if (!this.isDrawing) return;
      this.canvasContext.globalCompositeOperation = 'source-over';
      this.canvasContext.strokeStyle = this.tools[this.selectedToolIdx].color;
      this.canvasContext.beginPath();
      this.canvasContext.moveTo(this.lastX, this.lastY);
      this.canvasContext.lineTo(event.offsetX, event.offsetY);
      this.canvasContext.stroke();
      [this.lastX, this.lastY] = [event.offsetX, event.offsetY];
    },
    drawCursor(event) {
      this.cursorContext.beginPath();
      this.cursorContext.ellipse(
        event.offsetX, event.offsetY,
        this.brushSize, this.brushSize,
        0, 0, 0
      );
      this.cursorContext.stroke();
      setTimeout(() => {
        this.cursorContext.clearRect(0, 0, this.width, this.height);
      }, 100);
    }
  }
}
</script>
<style scoped>
.canvas-wrapper {
  position: relative;
}
#canvas {
  background-color: #f9f9f9;
  z-index: 0;
}
#cursor {
  pointer-events: none;
}
.draw-area canvas {
  position: absolute;
  left: 0;
  top: 0;
  border: 2px solid #c8c8c8;
  border-radius: 10px 10px 10px 10px;
}
</style>
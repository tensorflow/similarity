<template>
  <div class="Emoji">
    <h3>Emoji</h3>
    <div class=row>
      <div class="column">
        <drawingboard/>
        <button class="btn" v-on:click="submit">Submit</button>
      </div>
      <div class="column">Upload file</div>
    </div>
  </div>
</template>

<script>
import Drawingboard from "../components/Drawingboard.vue";
import axios from 'axios'

export default {
  name: "emoji",
  components: {
    Drawingboard
  },
  props: { 
    files: Array,
    neighbors: Array,
    loaded: Boolean,
    predicted_label: null,
    explain_src: String,
    neighbor_explain_srcs: Array,
    original_img_src: String,
    dataset: String("emoji"),
    show_image_explain: Boolean,
    show_target_explains: Boolean,
    num_targets_shown: Number,
    
  },
  methods: {
    submit: function() {
      console.log("Submit emoji")
      var c = document.getElementById("canvas")
      var ctx = c.getContext("2d")
      var imgData = ctx.getImageData(0, 0, 400, 400)
      var payload = { data: imgData.data, dataset: "emoji" }
      var path = 'http://localhost:5000/distances'
      console.log(payload, path)
      axios.post(path, payload).then(
        response => {
          this.$props = response.data

        }
      )
    }
  }
};
</script>

<style>
* {
  box-sizing: border-box;
}

.row {
  display: flex;
  justify-content: center;
}

.column {
  flex: 50%;
  padding: 10px;
  height: 100%;
  text-align: center;
  display: flex;
  justify-content: center;
}

.btn {
  transition-duration: 0.4s;
  height: 40px;
  width: 90px;
  background-color: #425066;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  border: none;
  color: #fff;
  border-radius: 30px;
}

.btn:hover {
  background-color: #FF6F00;
  color: #425066;
  border: none;
}
</style>
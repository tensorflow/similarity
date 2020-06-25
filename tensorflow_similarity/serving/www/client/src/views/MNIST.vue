<template>
  <div class="MNIST">
    <h3>MNIST</h3>
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
import axios from 'axios';

export default {
  name: "MNIST",
  components: {
    Drawingboard
  },
  data: function() {
    return {
      files: [],
                neighbors: [],
                loaded: false,
                predicted_label: null,
                explain_src: "",
                neighbor_explain_srcs: [],
                original_img_src: "",
                dataset: "mnist",
                show_image_explain: false,
                show_target_explains: false,
                num_targets_shown: 1
    }
  },
  methods: {
    submit: function() {
      console.log("Submit mnist")
      var c = document.getElementById("canvas")
      var ctx = c.getContext("2d")
      var imgData = ctx.getImageData(0, 0, 240, 240)
      var payload = { data: imgData.data, dataset: "mnist" }
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
  align-content: center;
}

.column {
  flex: 50%;
  padding: 10px;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
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
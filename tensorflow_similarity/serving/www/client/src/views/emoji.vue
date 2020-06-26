<template>
  <div class="Emoji">
    <h3>Emoji</h3>
    <div class=row>
      <div class="column-left">
        <drawingboard/>
        
      </div>
      <div class="column-right"><upload/></div>
    </div>
    <div class="row">
      <div class="column-left"><button class="btn" v-on:click="submit">Submit</button></div>
    </div>
    <div v-if="this.loaded">{{this.predicted_label}}</div>
  </div>
</template>

<script>
import Drawingboard from "../components/Drawingboard.vue";
import Upload from "../components/Upload.vue";
import axios from 'axios';

export default {
  name: "emoji",
  components: {
    Drawingboard,
    Upload
  },
  props: {
    files: []
  },
  data: function() {
    return {
      neighbors: [],
      loaded: false,
      predicted_label: null,
      explain_src: "",
      neighbor_explain_srcs: [],
      original_img_src: "",
      dataset: "emoji",
      how_image_explain: false,
      show_target_explains: false,
      num_targets_shown: 1
    }
  },
  methods: {
    submit: function() {
      console.log("Submit emoji")
      var c = document.getElementById("canvas")
      var ctx = c.getContext("2d")
      var imgData = ctx.getImageData(0, 0, 240, 240)
      var payload = { data: imgData.data, dataset: "emoji" }
      var path = 'http://localhost:5000/distances'
      console.log(payload, path)
      axios.post(path, payload).then(
        response => {
          console.log(response)
          this.data = response.data
          this.neighbors = this.data.neighbors
          this.loaded = true
          this.predicted_label = this.data.predicted_label
          this.explain_src = this.data.explain_src
          this.neighbor_explain_srcs = this.data.neighbor_explain_srcs
          this.original_img_src = this.data.original_img_src

        }
      )
    },
    getDistances: function(file) {
      var reader = new FileReader()
      reader.addEventListener("load", function () {
        var path = 'http://localhost:5000/distances'
        var payload = { data: reader.result, dataset: "emoji" }
        axios.post(path, payload).then(
        response => {                  
          this.data = response.data
          this.neighbors = this.data.neighbors
          this.loaded = true
          this.predicted_label = this.data.predicted_label
          this.explain_src = this.data.explain_src
          this.neighbor_explain_srcs = this.data.neighbor_explain_srcs
        })
      })

      reader.readAsDataURL(file)
    },
    watch: {
      files: function (val) {
        var file = null
        for (var index = 0; index < this.$props.files.length; index++) {
          file = val[index]
          var file_url = URL.createObjectURL(file)
          this.files[index].url = file_url
        }
        if (file !== null) {
          this.original_img_src = file_url
          this.getDistances(file)
        }
      }
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

.column-right {
  padding: 10px;
  height: 100%;
  display: flex;
  justify-content: left;
  align-items: center;
}

.column-left {
  padding: 10px;
  height: 100%;
  display: flex;
  
  flex-direction: column;
  justify-content: right;
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
  padding: 10px;
}

.btn:hover {
  background-color: #FF6F00;
  color: #425066;
  border: none;
}
</style>
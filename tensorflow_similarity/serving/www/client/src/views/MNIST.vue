<template>
  <div class="MNIST">
    <h3>MNIST</h3>
    <div class="row">
      <div class="column-left">
        <drawingboard/>
        
      </div>
      <div class="column-right"><upload @newFileUploaded="onNewUpload"/></div>
    </div>
    <div class="row">
      <div class="column-left"><button class="btn" v-on:click="submit">Submit</button></div>
    </div>
    <div v-if="this.loaded">
      <targets />
      <div class="row">
        <ul>
          <li v-for="(neighbor) in neighbors" v-bind:key="neighbor.label">
            <div class="card" v-bind:style="[neighbor.label === predicted_label ? {'background-color': '#FF8200'} : {'background-color': '#f6f6f6'}]">
                <div class=" card-image">
                  <figure class="image">
                    <img :src="`http://localhost:5000/static/images/${dataset}_targets/${neighbor.label}.png`">
                  </figure>
                </div>
              <div class="card-content">
                <div class="media">
                  <div class="media-content">
                    <p >{{"Label: " + neighbor.label}}</p>
                    <p >{{"Distance: " + neighbor.distance}}</p>
                  </div>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
import Drawingboard from "../components/Drawingboard.vue";
import Upload from "../components/Upload.vue";
import Targets from "../components/Targets.vue";
import axios from 'axios';

export default {
  name: "MNIST",
  components: {
    Drawingboard,
    Upload,
    Targets
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
      how_image_explain: false,
      show_target_explains: false,
      num_targets_shown: 1
    }
  },
  methods: {
    submit: function() {
      var c = document.getElementById("canvas")
      var ctx = c.getContext("2d")
      var imgData = ctx.getImageData(0, 0, 240, 240)
      var payload = { data: imgData.data, dataset: "mnist" }
      var path = 'http://localhost:5000/distances'
      axios.post(path, payload).then(
        response => {
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
    onNewUpload: function(val) {
      this.files.push(val)
      var file = null
      for (var index = 0; index < this.files.length; index++) {
        file = val[index]
        var file_url = URL.createObjectURL(file)
        this.files[index].url = file_url
      }

      if (file !== null) {
        this.original_img_src = file_url
        this.getDistances(file)
      }

    },
    getDistances: function(file) {
      var reader = new FileReader()
      reader.addEventListener("load", () => {
        var path = 'http://localhost:5000/distances'
        var payload = { data: reader.result, dataset: "mnist" }
        axios.post(path, payload).then(
          response => {    
            this.data = response.data
            this.neighbors = this.data.neighbors
            this.loaded = true
            this.predicted_label = this.data.predicted_label
            this.explain_src = this.data.explain_src
            this.neighbor_explain_srcs = this.data.neighbor_explain_srcs
            this.original_img_src = this.data.original_img_src  
          }
        )
      })
      reader.readAsDataURL(file)
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
  flex-direction: row;
}

.column-right {
  padding: 10px;
  height: 100%;
  display: flex;
  justify-content: left;
  align-items: center;
}

ul {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
}

.card-image {
  display: flex;
  justify-content: center;
}

.card {
  margin: 10px;
  padding: 10px;
  border-radius: 5px;
  color: #425066;
  text-align: center;
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

<template>
  <div class="emoji">
    <h3>Emoji</h3>
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
      var c = document.getElementById("canvas")
      var ctx = c.getContext("2d")
      var imgData = ctx.getImageData(0, 0, 240, 240)
      var payload = { data: imgData.data, dataset: "emoji" }
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
  }
};
</script>
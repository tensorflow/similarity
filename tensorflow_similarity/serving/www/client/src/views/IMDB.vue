<template>
  <div class="IMDB">
    <h3>IMDB</h3>
    <div class=row>
      <div class="column-left"><textarea id= "txtarea" v-model="text" placeholder="Type a movie review!"></textarea></div>
      <div class="column-right"><upload @newFileUploaded="onNewUpload"/></div>
    </div>
     <div class="row">
      <div class="column-left"><button class="btn" v-on:click="submit">Submit</button></div>
    </div>
    <div class="target-wrapper" v-if="this.loaded">
      <h3>Predicted Sentiment: {{this.predicted_label == 0 ? "Negative" : "Positive"}}</h3>
      <div class="row">
        <div class="scroll-menu-wrapper">
          <ul class="scroll-menu">
            <li v-for="(neighbor) in neighbors" v-bind:key="neighbor.label">
              <div class="card" v-bind:style="[neighbor.label === predicted_label ? {'background-color': '#FF8200'} : {'background-color': '#f6f6f6'}]">
                <div class="card-content">
                  <div class="media">
                    <div class="media-content">
                      <p>{{neighbor.label == 0 ? "Negative" : "Positive"}}</p>
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
  </div>
</template>

<script>
import Upload from "../components/Upload.vue";
import axios from 'axios';

export default {
  name: "IMDB",
  components: {
    Upload
  },
  data: function() {
    return {
      text: "",
      files: [],
      neighbors: [],
      loaded: false,
      predicted_label: null,
      explain_src: "",
      neighbor_explain_srcs: [],
      original_img_src: "",
      dataset: "imdb",
      how_image_explain: false,
      show_target_explains: false,
      num_targets_shown: 1
    }
  },
  methods: {
    submit: function() {
      var payload = { data: this.text, dataset: "imdb" }

      // currently the behavior for imdb is not implemented on the backend.
      // instead it simply return an empty response object.
      var path = 'http://localhost:5000/distances'
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
        var payload = { data: reader.result, dataset: "imdb" }
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

#txtarea {
  padding: 15px;
  width: 500px;
  height: 250px;
  font-size: 16px;
  background-color: #FFF;
  border: 1px solid #425066;
}
</style>

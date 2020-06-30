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
    <div v-if="this.loaded">{{this.predicted_label}}</div>
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
    onNewUpload(files) {
      console.log(files)
    },
    getDistances: function(file) {
      var reader = new FileReader()
      reader.addEventListener("load", function () {
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
        })
      })

      reader.readAsDataURL(file)
    },
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

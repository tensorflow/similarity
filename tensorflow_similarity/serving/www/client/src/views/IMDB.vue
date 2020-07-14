<template>
  <div class="IMDB">
    <h3></h3>
    <div class="row-top" >
      <div class="column-left"><textarea id= "txtarea" v-model="text" placeholder="Type a movie review!"></textarea></div>
      <div class="column-right"><upload @newFileUploaded="onNewUpload"/></div>
    </div>
     <div class="row">
      <div class="column-left"><button class="btn" v-on:click="submit">Submit</button></div>
    </div>
    <div class="target-wrapper" v-if="this.loaded">
      <div class="original-text-wrapper" v-if="this.uploaded_file">
        <p>{{this.original_text}}</p>
      </div>
      <h3 v-bind:style="{'font-weight': 'bold'}">Predicted Sentiment: {{this.predicted_label == 0 ? "Negative" : "Positive"}}</h3>
      <div class="row">
        <div class="scroll-menu-wrapper">
          <ul class="scroll-menu">
            <li v-for="(neighbor, index) in neighbors" v-bind:key="neighbor.label">
              <div class="card" v-bind:style="[neighbor.label === predicted_label ? {'background-color': '#FF8200'} : {'background-color': '#f6f6f6'}]">
                <div class="card-content">
                  <div class="media">
                    <div class="media-content">
                      <p>{{neighbor.label == 0 ? "Negative" : "Positive"}}</p>
                      <p >{{"Distance: " + neighbor.distance}}</p>
                      <button class="btn-embedding" v-on:click="mouseOver(index)">Show Text</button>
                    </div>
                  </div>
                </div>
              </div>
            </li>
          </ul>
        </div>
      </div>
      <p v-if="active" class="original-text-wrapper">{{neighbors[this.index].text}}</p>
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
      uploaded_file: false,
      neighbors: [],
      loaded: false,
      predicted_label: null,
      explain_src: "",
      neighbor_explain_srcs: [],
      original_text: "",
      dataset: "imdb",
      how_image_explain: false,
      show_target_explains: false,
      num_targets_shown: 1,
      active: false,
      index: -1
    }
  },
  methods: {
    mouseOver: function(index) {
      this.active = !this.active
      this.index = index
    },
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
          this.original_text = this.data.original_text 

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
        this.original_text = file_url
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
            this.original_text = this.data.original_text 
            this.uploaded_file = true
          }
        )
      })
      reader.readAsText(file)
    }
  }
};
</script>

<style>

.original-text-wrapper {
  font-style: italic; 
  width: 25%;
  text-align: left;
  word-wrap: break-word;
  display: inline-block;
  overflow: scroll;
  height: 150px;
  border-style: solid;
  border-radius: 5px;
  border-width: 1px;
  padding: 5px;
  margin: 10px;
}

#txtarea {
  padding: 15px;
  width: 500px;
  height: 195px;
  font-size: 16px;
  background-color: #FFF;
  border: 1px solid #425066;
}

.btn-embedding {
  transition-duration: 0.4s;
  height: 30px;
  width: 90px;
  margin-top: 10px;
  background-color: #425066;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  border: none;
  color: #fff;
  border-radius: 30px;
}


</style>

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
      console.log("Submit mnist")
      var c = document.getElementById("canvas")
      var ctx = c.getContext("2d")
      var imgData = ctx.getImageData(0, 0, 240, 240)
      var payload = { data: imgData.data, dataset: "mnist" }
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
    onNewUpload(files) {
      console.log('Upload received', files);
    },
    getDistances: function(file) {
      var reader = new FileReader()
      reader.addEventListener("load", function () {
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
        })
      })

      reader.readAsDataURL(file)
    },
    
    watch: {
      files: {
        deep: true,
        handler(val) {
          
          console.log("Uploaded file")
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
  align-items: right;
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

#txtarea {
  padding: 15px;
  width: 500px;
  height: 250px;
  font-size: 16px;
  background-color: #FFF;
  border: 1px solid #425066;
}
</style>
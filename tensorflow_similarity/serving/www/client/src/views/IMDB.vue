<template>
  <div class="IMDB">
    <h3>IMDB</h3>
    <div class=row>
      <div class="column"><input type="text" v-model="text"/></div>
      <div class="column">Upload file</div>
    </div>
  </div>
</template>

<script>

export default {
  name: "IMDB",
  components: {
    
  },
  data() {
    return {
      text: ""
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
</style>

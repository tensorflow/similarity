<template>
  <div id="app">
    <header>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdn.materialdesignicons.com/2.5.94/css/materialdesignicons.min.css">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.8.0/css/bulma.min.css">
        <link rel="stylesheet" href="https://unpkg.com/buefy/dist/buefy.min.css">
      <Navbar/>
      
    </header>
    <router-view />
  </div>
</template>



<script>
  import Navbar from './components/Navbar.vue';
  export default {
    name: 'app',
    components: {
      Navbar
    }
  }
</script>

<style>


#app {
  font-family: sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  box-sizing: border-box;
}

header {
  width: 100vw;
  background-color: #c4c4c4;
  padding: 5px;
  text-align: center;
}

</style>

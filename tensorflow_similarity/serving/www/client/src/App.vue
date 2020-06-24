<template>
  <div id="app">
    <header>
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
  background-color: #CCCCCC;
  padding: 5px;
  text-align: center;
}

</style>

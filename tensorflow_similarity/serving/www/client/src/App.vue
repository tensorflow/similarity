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
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

header {
  width: 100vw;
  background-color: #FF6F00;
  padding: 15px;
  text-align: center;
}

</style>

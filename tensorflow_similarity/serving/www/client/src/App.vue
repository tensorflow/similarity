<template>
  <div id="app">
    <div id="nav">
      <router-link to="/">MNIST</router-link> |
      <router-link to="/emoji">emoji</router-link>
      <router-link to="/pokemon">Pokemon</router-link>
      <router-link to="/IMDB">IMDB</router-link>
    </div>
    <router-view />
  </div>
</template>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  padding: 30px;
  margin: 10px;
}

#nav a {
  font-weight: bold;
  color: #2c3e50;
}

#nav a.router-link-exact-active {
  color: #42b983;
}
</style>

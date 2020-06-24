<template>
    <nav>
        <div class=logo></div>
        <div class="navbar-logo"><img alt="TF similarity logo" src="../assets/logo.svg" /></div>
        <div class="navbar-item"><router-link to="/">MNIST</router-link></div>
        <div class="navbar-item"><router-link to="/emoji">emoji</router-link></div>
        <div class="navbar-item"><router-link to="/pokemon">Pokemon</router-link></div>
        <div class="navbar-item"><router-link to="/IMDB">IMDB</router-link></div>   
        <div class="navbar-item"><a href="#"></a>Documentation</div>     

    </nav>
</template>

<script>

export default {
    name: 'navbar'
}
</script>

<style>
logo {
    
}
nav {
    display: flex;
    align-items: center;
    justify-content: right;
}

nav a {
    font-family: sans-serif;
    color: #E6E6E6;
    padding: 10px 20px;
    position: relative;
    text-align: center;
    display: flex;
    border-bottom: 3px solid transparent;
}


nav a.router-link-exact-active {
  color: #616161;
}

nav .navbar-item {
    color: #E6E6E6;
    padding: 10px 20px;
    position: relative;
    text-align: center;
    display: flex;
    border-bottom: 3px solid transparent;
}

nav .navbar-logo {
    padding: 10px 20px;
    position: relative;
    float:right;
}

</style>
<template>
    <nav>
        <div class="navbar-item"><img alt="TF similarity logo" src="../assets/logo.svg" /></div>
        <div class="navbar-item"><router-link to="/Custom">Try Your Model</router-link></div>  
        <div class="navbar-item"><router-link to="/">MNIST</router-link></div>
        <div class="navbar-item"><router-link to="/emoji">Emoji</router-link></div>
        <div class="navbar-item"><router-link to="/IMDB">IMDB</router-link></div> 

        <div class="navbar-item"><a href="https://github.com/tensorflow/similarity">Documentation</a></div>     

    </nav>
</template>

<script>
export default {
    name: 'navbar'
}
</script>

<style>

nav {
    display: flex;
    align-items: center;
    justify-content: right;
    width:100vw;
}

nav a {
    font-family: Roboto;
    color: #FFFFFF;
    padding: 5px 20px;
    position: relative;
    text-align: center;
    text-decoration: none;
    font-size: 20px;
    display: flex;
    border-bottom: 3px solid transparent;
    transition: 0.4s;
}

nav a.router-link-exact-active,
nav a:hover {
    color: #FF6F00;
    border-bottom-color: #FF6F00;
}

nav .navbar-item {
    padding: 5px 20px;
    font-size: 20px;
    position: relative;
    text-align: center;
    display: flex;
    text-decoration: none;
}

nav img {
    height: 100px;
    width: 250px;
}

</style>
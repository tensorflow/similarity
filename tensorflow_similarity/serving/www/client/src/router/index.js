import Vue from "vue";
import VueRouter from "vue-router";
import MNIST from "../views/MNIST.vue";
import emoji from "../views/emoji.vue";
import Pokemon from "../views/Pokemon.vue";
import IMDB from "../views/IMDB.vue";
import Custom from "../views/Custom.vue";


Vue.use(VueRouter);

const routes = [
  {
    path: "/",
    name: "MNIST",
    component: MNIST
  },
  {
    path: "/emoji",
    name: "emoji",
    component: emoji
  },
  {
    path: "/Pokemon",
    name: "Pokemon",
    component: Pokemon
  },
  {
    path: "/IMDB",
    name: "IMDB",
    component: IMDB
  },
  {
    path: "/Custom",
    name: "Custom",
    component: Custom
  }
];

const router = new VueRouter({
  mode: "history",
  base: process.env.BASE_URL,
  routes
});

export default router;

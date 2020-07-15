import Vue from "vue";
import VueRouter from "vue-router";
import MNIST from "../views/MNIST.vue";
import emoji from "../views/emoji.vue";
import IMDB from "../views/IMDB.vue";
import Custom from "../views/Custom.vue";


Vue.use(VueRouter);

const routes = [
  {
    path: "/",
    name: "MNIST",
    component: MNIST,
    meta: {
      title: "Tensorflow Similarity -  MNIST"
    }
  },
  {
    path: "/emoji",
    name: "emoji",
    component: emoji,
    meta: {
      title: "Tensorflow Similarity -  Emoji"
    }
  },

  {
    path: "/IMDB",
    name: "IMDB",
    component: IMDB,
    meta: {
      title: "Tensorflow Similarity -  IMDB"
    }
  },
  {
    path: "/Custom",
    name: "Custom",
    component: Custom,
    meta: {
      title: "Tensorflow Similarity - Custom"
    }
  }
];

const router = new VueRouter({
  mode: "history",
  base: process.env.BASE_URL,
  routes
});

const DEFAULT_TITLE = 'Tensorflow Similarity';
router.afterEach((to) => {
    Vue.nextTick(() => {
        document.title = to.meta.title || DEFAULT_TITLE;
    });
});

export default router;

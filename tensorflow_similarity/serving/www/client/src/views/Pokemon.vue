<template>
  <div class="Pokemon">
    <h3>Pokemon</h3>
    <div class=row>
      <div class="column">
        <drawingboard/>
        <button class="btn" v-on:click="submit">Submit</button>
      </div>
      <div class="column">Upload file</div>
    </div>
  </div>
</template>

<script>
import Drawingboard from "../components/Drawingboard.vue";

export default {
  name: "Pokemon",
  components: {
    Drawingboard
  },
  methods: {
    submit: function() {
      console.log("Submit Pokemon")
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
}

.btn:hover {
  background-color: #FF6F00;
  color: #425066;
  border: none;
}
</style>

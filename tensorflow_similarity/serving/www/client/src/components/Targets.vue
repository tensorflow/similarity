<template>
    <div class="Targets">
        <div class=row-t>
            <div class="column-t">
                <h3>Predicted Label: {{this.$parent.predicted_label}}</h3>
                <div class="original">
                    <img :src="this.$parent.original_img_src">
                </div>
            </div>
        </div>
    </div>
</template>

<script>

export default {
    name: "Targets"
}
</script>

<style>
.column-t {
  padding: 10px;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
}

.row-t {
  display: flex;
  justify-content: center;
  align-content: center;
}

.image {
    height: 50px;
    width: 50px;
    display: flex;
    justify-content: center;
    align-items: center;
}

img {
    border-radius: 5px;
}

.original {
    width: 200px;
    text-align: center;
}
</style>
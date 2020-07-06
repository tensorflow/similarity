import IMDB from "@/views/IMDB.vue"
import {shallowMount} from "@vue/test-utils"

describe("IMDB.vue", () => {
    let wrapper;
    beforeEach(() => {
        wrapper = shallowMount(IMDB, {
        })
    })

    it("renders", () => {
        expect(wrapper.exists()).toBe(true);
    })
})
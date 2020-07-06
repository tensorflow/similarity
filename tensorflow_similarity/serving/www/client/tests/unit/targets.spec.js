import Targets from "@/components/Targets.vue"
import {shallowMount} from "@vue/test-utils"

describe("Targets.vue", () => {
    let wrapper;
    beforeEach(() => {
        wrapper = shallowMount(Targets, {
        })
    })

    it("renders", () => {
        expect(wrapper.exists()).toBe(true);
    })

    it("has predicted label", () => {
        expect(wrapper.html()).toContain('<h3>Predicted Label: </h3>')
    })

    it("has target image", () => {
        expect(wrapper.html()).toContain('<div class=\"original\"><img></div>')
    })
})
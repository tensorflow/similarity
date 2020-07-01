import Custom from "@/views/Custom.vue"
import {shallowMount} from "@vue/test-utils"

describe("Custom.vue", () => {
    let wrapper;
    beforeEach(() => {
        wrapper = shallowMount(Custom, {
        })
    })

    it("renders", () => {
        expect(wrapper.exists()).toBe(true);
    })

    it("has correct header", () => {
        expect(wrapper.html()).toContain('<h3>Try your own Model!</h3>')
    })

})
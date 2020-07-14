import Navbar from "@/components/Navbar.vue"
import {shallowMount} from "@vue/test-utils"

describe("Navbar.vue", () => {
    let wrapper;
    beforeEach(() => {
        wrapper = shallowMount(Navbar, {
            stubs: ['router-link', 'router-view']
        })
    })

    it("renders", () => {
        expect(wrapper.exists()).toBe(true);
    })

    it("has router link to custom model", () => {
        expect(wrapper.html()).toContain('<router-link-stub to=\"/Custom\">Try Your Model</router-link-stub>')
    })

    it("has router link to MNIST model", () => {
        expect(wrapper.html()).toContain('<router-link-stub to=\"/\">MNIST</router-link-stub>')
    })

    it("has router link to emoji model", () => {
        expect(wrapper.html()).toContain('<router-link-stub to=\"/emoji\">Emoji</router-link-stub>')
    })

    it("has router link to imdb model", () => {
        expect(wrapper.html()).toContain('<router-link-stub to=\"/IMDB\">IMDB</router-link-stub>')
    })

    it("has link to documentation", () => {
        expect(wrapper.html()).toContain('<a href=\"https://github.com/tensorflow/similarity\">Documentation</a>')
    })
})
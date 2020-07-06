import Emoji from "@/views/emoji.vue"
import {shallowMount} from "@vue/test-utils"

describe("Emoji.vue", () => {
    let wrapper;
    beforeEach(() => {
        wrapper = shallowMount(Emoji, {
            data() {
                return {
                    files: [],
                    neighbors: [],
                    loaded: false,
                    predicted_label: null,
                    explain_src: "",
                    neighbor_explain_srcs: [],
                    original_img_src: "",
                    dataset: "emoji",
                    how_image_explain: false,
                    show_target_explains: false,
                    num_targets_shown: 1
                }
            }
        })
    })

    it("renders", () => {
        expect(wrapper.exists()).toBe(true);
    })

    it("has drawing board", () => {
        expect(wrapper.html()).toContain('<drawingboard-stub brushsize=\"12\" width=\"240\" height=\"240\"></drawingboard-stub>')
    })

    it("has upload component", () => {
        expect(wrapper.html()).toContain('<upload-stub></upload-stub>')
    })

    it("doesn't show cards if cards have not loaded", () => {
        expect(wrapper.find('.target-wrapper').exists()).toBeFalsy();
    })

})

test("cards are displayed once the response from the backend is received", () => {
    let wrapper = shallowMount(Emoji, {
        data() {
            return {
                files: [],
                neighbors: [],
                loaded: true,
                predicted_label: null,
                explain_src: "",
                neighbor_explain_srcs: [],
                original_img_src: "",
                dataset: "emoji",
                how_image_explain: false,
                show_target_explains: false,
                num_targets_shown: 1
            }
        }
    })

    expect(wrapper.find('.target-wrapper').exists()).toBeTruthy();
})

jest.mock('axios', () => ({
    post: Promise.resolve(
        {
            neighbors: ["neighbor1"],
            loaded: true,
            predicted_label: "1",
            explain_src: "image_file",
            neighbor_explain_srcs: ["image_file"],
            original_img_src: "image_file"
        }
    )
}))


it("renders results when response is received from backend", async () => {
    let wrapper = shallowMount(Emoji, {
        data() {
            return {
                files: [],
                neighbors: [],
                loaded: false,
                predicted_label: null,
                explain_src: "",
                neighbor_explain_srcs: [],
                original_img_src: "",
                dataset: "emoji",
                how_image_explain: false,
                show_target_explains: false,
                num_targets_shown: 1
            }
        }
    })
    wrapper.vm.getDistances(new Blob())
    wrapper.vm.$nextTick(() => {
        expect(wrapper.vm.loaded).toBeTruthy
    })
    
})

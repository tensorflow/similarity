import Upload from "@/components/Upload.vue"
import {shallowMount} from "@vue/test-utils"

let wrapper = shallowMount(Upload, {stubs: ['b-field', 'b-icon', 'b-upload']})
describe("Upload.vue", () => {
    it("renders", () => {
        expect(wrapper.exists()).toBe(true);
    })

    it("has upload vmodel", () => {
        expect(wrapper.html()).toContain('<b-upload-stub multiple=\"\" drag-drop=\"\" value=\"\">')
    })

    it('has delete button', () => {
        expect(wrapper.exists('button')).toBe(true)
    })

    it('delete button should delete files', () => {
        let files = wrapper.vm.dropFiles
        wrapper.vm.dropFiles.push("file")
        wrapper.vm.deleteDropFile(0)
        expect(wrapper.vm.dropFiles).toBe(files)
    })
})


test("emit file upload event when a new file is uploaded", async () => {
    wrapper.vm.dropFiles.push("file")

    await wrapper.vm.$nextTick()

    expect(wrapper.emitted().newFileUploaded).toBeTruthy()
})
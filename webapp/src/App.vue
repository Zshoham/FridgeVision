<template>
  <div class="app">
    <FormulateForm class="login-form" @submit="upload">
      <h1 class="form-text">Fridge Vision</h1>
      <p class="form-text">
        If this is your first time visiting our site please enter your a name,
        which will use for identification, and an initial image of your fridge.
      </p>
      <p class="form-text">
        Otherwise enter the name you provided last time, and a new picture of your fridge,
        we will tell you what groceries you are missing.
      </p>
      <FormulateInput
        name="username"
        type="text"
        label="Username"
        v-model="username"
        validation="required"/>
      <FormulateInput name="fridge" :uploader="updateFile" type="image" label="Image of Your Fridge" upload-behavior="delayed" validation="required"/>
      <div class="submit">
        <FormulateInput class="submit" type="submit" label="Send"/>
      </div>
      <p class="result-text" v-if="state == 'RECIVED'"> 
        {{this.result}}
      </p>
    </FormulateForm>
  </div>
</template>

<script>
import axios from "axios"

export default {
  name: 'App',
  data() {
    return {
      state: "IDLE",
      result: "PLACE HOLDER",
      username: "",
      file: null
    }
  },
  methods: {
    async upload() {
      this.state = "SENT"
      let form_data = new FormData()
      form_data.set("file", this.file)
      let response = await axios.post("http://localhost:5000/detect/" + this.username, form_data)
      this.result = response.data
      this.state = "RECIVED"
    },
    async updateFile(file, progress) {
      this.file = file
      progress(100)
      return Promise.resolve({})
    }
  }
}
</script>

<style scoped>

.app {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 5%;
}

.submit {
  display: flex;
  align-items: center;
  justify-content: center;
}

.login-form {
  padding: 2em;
  border: 1px solid #a8a8a8;
  border-radius: .5em;
  max-width: 500px;
  box-sizing: border-box;
}
.form-text {
  margin-top: 0;
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol
}

.result-text {
  margin-top: 3em;
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol
}

.login-form::v-deep .formulate-input .formulate-input-element {
  max-width: none;
}

</style>

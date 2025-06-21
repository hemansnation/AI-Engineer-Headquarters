import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:8000/token", new URLSearchParams({
        username: email,
        password: password,
      }));
      setToken(res.data.access_token);
      localStorage.setItem("token", res.data.access_token);

    } catch (error) {
      alert("Login failed. Please check your credentials.");
  }
};

const handleSunmit = async () => {
  try{
    const res = await axios.post(
      "http://127.0.0.1:8000/chat", {question}, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setResponse(res.data.response);
  } catch (error) {
    alert("Error fetching response. Please try again.");
  }
};

  return (
    <div className="App">
      {!token ? (
        <div>
          <h2>HR Query Bot Login</h2>
          <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
          <button onClick={handleLogin}>Login</button>
        </div>
      ) : (
        <div>
          <h2>HR Query Bot</h2>
          <textarea
            placeholder="Ask your HR-related question here..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            />
          <button onClick={handleSunmit}>Submit</button>
          <div>
            <h3>Response:</h3>
            <p>{response}</p>
            </div>
        </div>
      )}
    </div>
  );
}
export default App;

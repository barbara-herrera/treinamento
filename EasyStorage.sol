pragma solidity ^0.5.3;


contract EasyStorage {

  // Informações gerais do contrato
  address payable private contract_owner_address;
  string private contract_title;
  string private contract_description;

  uint private contract_price_to_rent_unit_storage; // 100 finney
  uint private contract_price_to_add_storage_owner; // 100 finney

  // Struct que representa as informações do aluguel de Storage
  struct Storage {
    address payable storage_owner_address;
    address payable storage_tenant_adrress;
	uint storage_units;
	
  }

  // Fornecendo um hash, eu identifico um contrato de armazenamento
  mapping(uint256 => Storage) private storage_list;
  
  // Lista para identificar os donos de armazens, somente os ativos podem ser donos de um local e alugar espaços
  mapping(address => bool) private storage_owner_address_list;
  

   constructor() public {
    contract_owner_address = msg.sender;
    contract_title = "Titulo do Contrato Aluguel de Armazenamento.";
    contract_description = "Contrato para aluguel de armazenamentos.";
    contract_price_to_rent_unit_storage = 100 finney;
	contract_price_to_add_storage_owner = 100 finney;
  }

  modifier isContractOwner {
    require(msg.sender == contract_owner_address, "Somente o dono do contrato pode chamar esta função.");
    _;
  }
  
  modifier isStorageOwner {
    require(storage_owner_address_list[msg.sender] == true, "Somente o dono do contrato pode chamar esta função.");
    _;
  }
  
  
  // Seta o titulo do contrato
  function setContractTitle(string memory _contract_title) public isContractOwner {
    contract_title = _contract_title;
  }

  // Seta a descrição do contrato
  function setContractDescription(string memory _contract_description) public isContractOwner {
    contract_description = _contract_description;
  }

  // Seta o endereço do dono deste contrato
  function setContractOwnerAddress(address payable _contract_owner_address) public isContractOwner {
    contract_owner_address = _contract_owner_address;
  }

  // Seta o preço para adicionar um novo dono de armazem. Cada dono de armazem deve adicionar  dono do espaço que irá disponibilizar para alugar
  function setContractPriceToAddStorageOwner(uint _contract_price_to_add_storage_owner) public isContractOwner {
    contract_price_to_add_storage_owner = _contract_price_to_add_storage_owner;
  }


  // Adiciona um aluguel de storage no contrato
  function addStorageOwner(uint256 storage_hash) public payable {
      require(storage_list[storage_hash].storage_owner_address == address(0), "Storage já possui dono");
      require(msg.value == contract_price_to_add_storage_owner, "Preço para adicionar um novo armazem incorreto");

      storage_list[storage_hash].storage_owner_address = msg.sender;
      
	  contract_owner_address.transfer(msg.value);
  }

  // Aluguel de uma storage unit uma storage
  function rentStorage(uint256 storage_hash, address payable _storage_owner, uint _storage_units) public payable {
    require(storage_list[storage_hash].storage_owner_address != address(0), "Storage não existe!");
    require(contract_price_to_rent_unit_storage == msg.value, "Valor de compra da unidade de storage inválido");
    
    storage_list[storage_hash].storage_owner_address = _storage_owner;
    storage_list[storage_hash].storage_units = _storage_units;
    
    storage_list[storage_hash].storage_owner_address.transfer(msg.value);

    storage_list[storage_hash].storage_tenant_adrress = msg.sender;
  }

  function getContractTitle() public view returns (string memory) {
    return contract_title;
  }

  function getContractDescription() public view returns (string memory) {
    return contract_description;
  }

  function getContractOwnerAddress() public view returns (address) {
    return contract_owner_address;
  }

  function getContractPriceToAddStorageOwner() public view returns (uint256) {
    return contract_price_to_add_storage_owner;
  }

  function getPriceToRentStorage() public view returns (uint) {
    return contract_price_to_rent_unit_storage;
  }

  function getStorageOwner(uint256 storage_hash) public view returns (address) {
    return storage_list[storage_hash].storage_owner_address;
  }

  function getStorageHistoric(uint256 storage_hash) public view returns (address) {
    return storage_list[storage_hash].storage_tenant_adrress;
  }

  function killContract() public isContractOwner() {
	contract_owner_address.transfer(address(this).balance);
	selfdestruct(contract_owner_address);
  }
}
